from __future__ import absolute_import

import torch
from torch import nn


class RangeLoss(nn.Module):
    """
        Range_loss = alpha * intra_class_loss + beta * inter_class_loss
        intra_class_loss is the harmonic mean value of the top_k largest distances beturn intra_class_pairs
        inter_class_loss is the shortest distance between different class centers
    """
    def __init__(self, k=2, margin=0.1, alpha=0.5, beta=0.5, use_gpu=True, ordered=True, ids_per_batch=32, imgs_per_id=4):
        super(RangeLoss, self).__init__()
        self.use_gpu = use_gpu
        self.margin = margin
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.ordered = ordered
        self.ids_per_batch = ids_per_batch
        self.imgs_per_id = imgs_per_id

    def _pairwise_distance(self, features):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
         Return: 
            pairwise distance matrix with shape(batch_size, batch_size)
        """
        n = features.size(0)
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, features, features.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _compute_top_k(self, features):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
         Return: 
            top_k largest distances
        """
        # reading the codes below can help understand better
        '''
        dist_array_2 = self._pairwise_distance(features)
        n = features.size(0)
        mask = torch.zeros(n, n)
        if self.use_gpu: mask=mask.cuda()
        for i in range(0, n):
            for j in range(i+1, n):
                mask[i, j] += 1
        dist_array_2 = dist_array_2 * mask
        dist_array_2 = dist_array_2.view(1, -1)
        dist_array_2 = dist_array_2[torch.gt(dist_array_2, 0)]
        top_k_2 = dist_array_2.sort()[0][-self.k:]
        print(top_k_2)
        '''
        dist_array = self._pairwise_distance(features)
        dist_array = dist_array.view(1, -1)
        top_k = dist_array.sort()[0][0, -self.k * 2::2]     # Because there are 2 same value of same feature pair in the dist_array
        # print('top k intra class dist:', top_k)
        return top_k

    def _compute_min_dist(self, center_features):
        """
         Args:
            center_features: center matrix (before softmax) with shape (center_number, center_dim)
         Return: 
            minimum center distance
        """
        '''
        # reading codes below can help understand better
        dist_array = self._pairwise_distance(center_features)
        n = center_features.size(0)
        mask = torch.zeros(n, n)
        if self.use_gpu: mask=mask.cuda()
        for i in range(0, n):
            for j in range(i + 1, n):
                mask[i, j] += 1
        dist_array *= mask
        dist_array = dist_array.view(1, -1)
        dist_array = dist_array[torch.gt(dist_array, 0)]
        min_inter_class_dist = dist_array.min()
        print(min_inter_class_dist)
        '''
        n = center_features.size(0)
        dist_array2 = self._pairwise_distance(center_features)
        min_inter_class_dist2 = dist_array2.view(1, -1).sort()[0][0][n]  # exclude self compare, the first one is the min_inter_class_dist
        return min_inter_class_dist2

    def _calculate_centers(self, features, targets, ordered, ids_per_batch, imgs_per_id):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
         Return: 
            center_features: center matrix (before softmax) with shape (center_number, center_dim)
        """
        if self.use_gpu:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]
                else:
                    unique_labels = targets.cpu().unique().cuda()
            else:
                unique_labels = targets.cpu().unique().cuda()
        else:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]
                else:
                    unique_labels = targets.unique()
            else:
                unique_labels = targets.unique()

        center_features = torch.zeros(unique_labels.size(0), features.size(1))
        if self.use_gpu:
            center_features = center_features.cuda()

        for i in range(unique_labels.size(0)):
            label = unique_labels[i]
            same_class_features = features[targets == label]
            center_features[i] = same_class_features.mean(dim=0)
        return center_features

    def _inter_class_loss(self, features, targets, ordered, ids_per_batch, imgs_per_id):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            margin: inter class ringe loss margin
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
         Return: 
            inter_class_loss
        """
        center_features = self._calculate_centers(features, targets, ordered, ids_per_batch, imgs_per_id)
        min_inter_class_center_distance = self._compute_min_dist(center_features)
        # print('min_inter_class_center_dist:', min_inter_class_center_distance)
        return torch.relu(self.margin - min_inter_class_center_distance)

    def _intra_class_loss(self, features, targets, ordered, ids_per_batch, imgs_per_id):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
         Return: 
            intra_class_loss
        """
        if self.use_gpu:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]
                else:
                    unique_labels = targets.cpu().unique().cuda()
            else:
                unique_labels = targets.cpu().unique().cuda()
        else:
            if ordered:
                if targets.size(0) == ids_per_batch * imgs_per_id:
                    unique_labels = targets[0:targets.size(0):imgs_per_id]
                else:
                    unique_labels = targets.unique()
            else:
                unique_labels = targets.unique()

        intra_distance = torch.zeros(unique_labels.size(0))
        if self.use_gpu:
            intra_distance = intra_distance.cuda()

        for i in range(unique_labels.size(0)):
            label = unique_labels[i]
            same_class_distances = 1.0 / self._compute_top_k(features[targets == label])
            intra_distance[i] = self.k / torch.sum(same_class_distances)
        # print('intra_distace:', intra_distance)
        return torch.sum(intra_distance)

    def _range_loss(self, features, targets, ordered, ids_per_batch, imgs_per_id):
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             range_loss
        """
        inter_class_loss = self._inter_class_loss(features, targets, ordered, ids_per_batch, imgs_per_id)
        intra_class_loss = self._intra_class_loss(features, targets, ordered, ids_per_batch, imgs_per_id)
        range_loss = self.alpha * intra_class_loss + self.beta * inter_class_loss
        return range_loss, intra_class_loss, inter_class_loss

    def forward(self, features, targets):
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             range_loss
        """
        assert features.size(0) == targets.size(0), "features.size(0) is not equal to targets.size(0)"
        if self.use_gpu:
            features = features.cuda()
            targets = targets.cuda()

        range_loss, intra_class_loss, inter_class_loss = self._range_loss(features, targets, self.ordered, self.ids_per_batch, self.imgs_per_id)
        return range_loss, intra_class_loss, inter_class_loss


if __name__ == '__main__':
    use_gpu = False
    range_loss = RangeLoss(use_gpu=use_gpu, ids_per_batch=4, imgs_per_id=4)
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).cuda()
    loss = range_loss(features, targets)
    print(loss)
