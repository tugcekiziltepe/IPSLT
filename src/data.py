import cv2
import random
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import gzip
import pickle
import torch.utils.data.dataset as Dataset
from PIL import Image
from src.utils import pad_or_truncate_frames


pad_embeddings = lambda a, i: a[0: i] if a.shape[0] > i else np.concatenate((a, np.zeros((i - a.shape[0], a.shape[1]))), axis = 0) # for BERT embeddings


PAD_IDX = 0
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class PhoenixDataset(Dataset.Dataset):
    def __init__(self,text_embeddings_path, 
                        features_path, 
                        pt_file_path,
                        texts_max_length,
                        features_max_length, transform=None):
        
        self.data = torch.load(pt_file_path)
        self.text_embeddings_path = text_embeddings_path
        self.features_path = features_path
        self.texts_max_length = texts_max_length
        self.features_max_length = features_max_length
        self.keys = list(self.data.keys()) 
        self.transform = transform
         

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        key = self.keys[idx]
        sample_data = self.data[key]

        text_embedding_path = os.path.join(self.text_embeddings_path, f"{key}.npy")
        if not os.path.exists(text_embedding_path):
            raise FileNotFoundError(f"Text embedding file not found: {text_embedding_path}")

        text_embeddings = np.load(text_embedding_path)
        text_length = text_embeddings.shape[0]
        text_embeddings = pad_embeddings(text_embeddings, self.texts_max_length).astype(np.float32)

        features_path = os.path.join(self.features_path, f"{key}.npy")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features path file not found: {features_path}")
        
        features = np.load(features_path)
        features_length = features.shape[0]
        truncated_features = pad_or_truncate_frames(features, self.features_max_length)
        truncated_features = truncated_features.astype(np.float32)

        sample = {
            'text': sample_data["text"],
            'text_embeddings': text_embeddings,
            'features': truncated_features,
            'gloss': sample_data["gloss"],
            'speaker': sample_data["speaker"],
            'features_length': features_length,
            'idx': idx,
            'name': key
        }
        if self.transform:
            sample = self.transform(sample)

        return sample
