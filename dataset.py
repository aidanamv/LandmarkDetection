import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import json


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=4096,
                 fold=1,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.fold = fold
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}
        self.datapath = []

        self.meta = {}
        splitfile = os.path.join(self.root, 'fold_{}'.format(self.fold), '{}_data.json'.format(split))
        # from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:

            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, uuid + '.npz')))

        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        data = np.load(fn[1])
        point_set = data["vertices"].astype(np.float32)
        # color_set = data[:, 3:6].astype(np.float32)
        seg = data["labels"].astype(int)
        # print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        # color_set = color_set[choice, :]
        trans = np.expand_dims(np.mean(point_set, axis=0), 0)
        point_set = point_set - trans  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        # point_set = np.concatenate([point_set, color_set], axis=1)
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    datapath = "../new_new_dataset"

    d = ShapeNetDataset(root=datapath, class_choice=['Spine'])
    ps, seg = d[0]

    d = ShapeNetDataset(root=datapath, classification=True)
    ps, cls = d[0]
    # get_segmentation_classes(datapath)
