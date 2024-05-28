import torch
from torch.utils.data import Dataset, DataLoader

class FewshotDataset(Dataset):
    def __init__(self, train_data, train_label, episode_num=1000, way_num=7, shot_num=1, query_num=1):
        self.train_data = train_data
        self.train_label = train_label
        self.episode_num = episode_num
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num

    def __len__(self):
        return self.episode_num

    def __getitem__(self, index):
        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        label_indices = torch.randperm(len(self.train_label)).cuda()
        train_label_gpu = self.train_label.cuda()
        train_data_gpu = self.train_data.cuda()

        for label_num in range(self.way_num):
            support_idxs = torch.nonzero(train_label_gpu[label_indices] == label_num, as_tuple=False).flatten()
            support_idxs = support_idxs[:self.shot_num]
            support_data = train_data_gpu[label_indices][support_idxs]

            query_idxs = torch.nonzero(train_label_gpu[label_indices] == label_num, as_tuple=False).flatten()
            query_idxs = query_idxs[~torch.isin(query_idxs, support_idxs)][:self.query_num]
            query_data = train_data_gpu[label_indices][query_idxs]
            query_data_targets = train_label_gpu[label_indices][query_idxs]

            query_images.append(query_data)
            query_targets.append(query_data_targets)
            support_images.append(support_data)
            support_targets.append(torch.full((self.shot_num,), label_num).cuda())

        query_images = torch.cat(query_images, dim=0)
        query_targets = torch.cat(query_targets, dim=0)
        support_images = torch.cat(support_images, dim=0)
        support_targets = torch.cat(support_targets, dim=0)

        return query_images, query_targets, support_images, support_targets



'''
  way_num = num_classes, shot_num = number samples per class

'''