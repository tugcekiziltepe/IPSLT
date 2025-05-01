import cv2
import random
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class PHEONIX14T(Dataset):

    def __init__(self, csv_path, videos_path, transforms, max_len):
        self.transforms = transforms
        self.videos_path = videos_path
        self.max_len = max_len
        self.names = []

        with open(csv_path, 'r') as file:
            all_lines = file.read().splitlines()

        headers = all_lines[0].split('|')

        self.csv_file = dict()

        for line in all_lines[1:]:
            items = line.split('|')
            line_dict = {header: item for header, item in zip(headers, items)}
            self.names.append(line_dict["name"])
            self.csv_file[line_dict["name"]] = line_dict


    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        img_paths = sorted(glob.glob(os.path.join(self.videos_path, name) + "/*.png"))


        if len(img_paths) > self.max_len:
            tmp = sorted(random.sample(range(len(img_paths)), k=self.max_len))
            new_paths = []
            for i in tmp:
                new_paths.append(img_paths[i])
            img_paths = new_paths

        imgs = []

        for path in img_paths:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) / 255.0
            img = cv2.resize(img, (224, 224))
            img = self.transforms(img)
            imgs.append(img)

        imgs = np.stack(imgs, axis=0)
        gloss = self.csv_file[name]["orth"]
        translation = self.csv_file[name]["translation"]

        return imgs, gloss, translation

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    videos, glosses, translations = zip(*batch)

    # Convert numpy arrays to tensors if they aren't already
    # videos = [torch.tensor(video, dtype=torch.float).permute(0, 2, 3, 1) if isinstance(video, np.ndarray) else video for video in videos]
    videos = [torch.tensor(video, dtype=torch.float).permute(0, 2, 3, 1) if isinstance(video, np.ndarray) else video for video in videos]
    max_length = max(video.shape[0] for video in videos)
    padded_videos = []

    for video in videos:
        pad_size = max_length - video.shape[0]
        # Use PyTorch operations for padding
        padded_video = torch.cat([video, video[-1].unsqueeze(0).repeat(pad_size, 1, 1, 1)], dim=0)
        padded_videos.append(padded_video)

    padded_videos = torch.stack(padded_videos)

    translations = tokenizer(translations, return_tensors="pt", padding=True, truncation=True)["input_ids"]

    return padded_videos, translations

