import os
import random
from glob import glob
from glog import logger
from torch.utils.data import Dataset
from data import aug
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import re

MAX_FRAMES = 5  # Maximum number of LR frames loaded per sample (padded to this count)

class LRHRDataset(Dataset):
    def __init__(self, opt, phase):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        height = opt['height']
        width = opt['width']

        self.files_lr = os.path.join(opt['dataroot'], phase, 'inputs')
        self.files_hr = os.path.join(opt['dataroot'], phase, 'gt')

        lr_folders = os.listdir(self.files_lr)
        self.lr = []
        self.hr = []

        for folder in lr_folders:
            self.lr.append(sorted(glob(os.path.join(self.files_lr, folder, '*.jpg')), key=self.extract_number))
            self.hr.extend(glob(os.path.join(self.files_hr, folder, '*.jpg')))


        assert len(self.lr) == len(self.hr)

        # Limit dataset size if data_len is specified
        data_len = opt.get('data_len', -1)
        if data_len is not None and data_len > 0:
            self.lr = self.lr[:data_len]
            self.hr = self.hr[:data_len]

        self.lr_transform_fn = aug.get_transforms(size=(height, width))
        self.transform_fn = aug.get_transforms(size=(height, width))

        self.normalize_fn = aug.get_normalize()
        logger.info(f'Dataset has been created with {len(self.lr)} samples')
        
    def extract_number(self, file_path):
        match = re.search(r'img_(\d+).jpg', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        else:
            print('Sort Error at: ', file_path)
            return -1


    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        assert len(self.lr[idx]) != 0, f'No LR images for index {idx}.'
        n_available = len(self.lr[idx])
        n_to_sample = min(n_available, MAX_FRAMES)
        sampled_paths = sorted(random.sample(self.lr[idx], n_to_sample))

        # Pad to MAX_FRAMES by repeating the last frame so batches have uniform shape.
        # The actual number of unique frames used at training time is chosen randomly
        # in feed_data (1 to MAX_FRAMES), so the padded frames are discarded.
        while len(sampled_paths) < MAX_FRAMES:
            sampled_paths.append(sampled_paths[-1])

        lr_tensors = []
        for path in sampled_paths:
            img = np.array(Image.open(path))
            img = self.lr_transform_fn(img)
            img = self.normalize_fn(img)
            img = transforms.ToTensor()(img)
            lr_tensors.append(img)

        hr_image = np.array(Image.open(self.hr[idx]))
        hr_image = self.transform_fn(hr_image)
        hr_image = self.normalize_fn(hr_image)
        hr_image = transforms.ToTensor()(hr_image)

        lr_seq = torch.stack(lr_tensors, dim=0)  # (MAX_FRAMES, 3, H, W)
        return {'LR_seq': lr_seq, 'HR': hr_image, 'path': self.hr[idx]}

    def load_data(self):
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.num_threads))
        return dataloader


def create_dataset(opt):
    return LRHRDataset(opt).load_data()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default=r'')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    opt = args
    dataset = LRHRDataset(opt)
    data = dataset[0]
    print(data['LR_seq'].shape)  # expected: (MAX_FRAMES, 3, H, W)


