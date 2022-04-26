from torch.utils.data import Dataset, DataLoader
import torch

import pandas as pd
import os
from PIL import Image
import json
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, jsonfile, root, transforms = None):
        with open(os.path.join(root, jsonfile)) as f:
            self.json = json.load(f)

        self.root = root
        self.transforms = transforms

    def __len__(self):
        return len(self.json)

    def __getitem__(self, idx):
        name = self.json[idx]['file_name']
        im = Image.open(os.path.join(self.root, 'images', name)).convert('RGB')

        targets_json = self.json[idx]['bb']
        targets = [item['bbox'] for item in targets_json]

        if self.transforms is not None:
            im = self.transforms(im)

        return im, targets

def get_custom_train(config,
                        transform = None,
                        target_transform = None):
    dataset = CustomDataset(config['CSV'], config['ROOT'], transform)
    loader = get_dataloader(dataset, config['BATCH_SIZE'], config['SHUFFLE'], config['NUM_WORKERS'])

    return loader

def get_dataloader(dataset,
                   batch_size = 1,
                   shuffle = False,
                   num_workers = 0):
    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      shuffle = shuffle,
                      num_workers = num_workers,
                      collate_fn = detection_collate)

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return imgs, targets
