
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from transformers import AutoTokenizer
import argparse
import pytorch_lightning as pl
from src.utils import load_config
from src.data import PhoenixDataset
from src.model import IPSLT
import torchvision

import wandb
wandb.login(key="cf46c789616ecd669352aa388a452318708b615e")  

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="valid/total_loss",  # Stop if train loss doesn't improve
    mode="min", 
    patience=150,  # Stop training if no improvement for 150 epochs
    verbose=True
)

# Best checkpoint based on valid/pose_encodings_loss with epoch number
best_pose_checkpoint = ModelCheckpoint(
    monitor="valid/total_loss", # Choose which loss to monitor
    mode="min", # Save when this loss decreases
    save_top_k=1,  # Keeps only the best model
    filename="best-val-{epoch}",  # Includes epoch number
    verbose=False,
)

# Last checkpoint with epoch number
last_checkpoint = ModelCheckpoint(
    filename="last-checkpoint-epoch={epoch}",
    save_last=False,
    verbose=False,
    auto_insert_metric_name=False  # Prevents creating multiple checkpoints
)

def train(args, cfg, model, model_path, vocab, 
               train_dataloader, val_dataloader):
    
    model = model(
        cfg=cfg, 
        args=args,
        text_vocab=vocab
    )
     
    model_trainer = pl.Trainer(
    callbacks=[best_pose_checkpoint, last_checkpoint, early_stopping],
    devices=4,
    num_nodes=1, 
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true"
    )
    # model_trainer = pl.Trainer(
    #     accelerator="cpu",  # CPU kullanımı için bu satırı ekledik
    #     callbacks=[best_pose_checkpoint, last_checkpoint, early_stopping]
    # )
    model_trainer.fit(model, train_dataloader, val_dataloader)
    

args = argparse.Namespace(
    mode="train",
    config_path=r"configs/k3wkl15_vega.yaml",
    model_path = None,
    device='cuda',
    seed=42,
    resume=False,
    start_epoch=0,
    batch_size=64,
    epochs=379,
    num_workers=8,
    print_frequency=100
    )

cfg_file = args.config_path
cfg = load_config(cfg_file)

wandb.init(project="IPSLT", name="run1_k3wkl3", config=cfg) 
device = "cuda" if torch.cuda.is_available() else "cpu"

pl.seed_everything(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

random.seed(args.seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
device = "cuda"


train_embeddings_path = cfg['data']['train']['embeddings_path']
train_features_path = cfg['data']['train']['features_path']
train_pt_file_path = cfg['data']['train'].get('pt_file_path', None)

val_embeddings_path = cfg['data']['dev']['embeddings_path']
val_features_path = cfg['data']['dev']['features_path']
val_pt_file_path = cfg['data']['dev'].get('pt_file_path', None)

train_dataset = PhoenixDataset(text_embeddings_path=train_embeddings_path, 
                        features_path=train_features_path, 
                        pt_file_path=train_pt_file_path,
                        texts_max_length=cfg['data']['max_text_len'],
                        features_max_length=cfg['data']['max_frame_len'])

val_dataset = PhoenixDataset(text_embeddings_path=val_embeddings_path, 
                        features_path=val_features_path, 
                        pt_file_path=val_pt_file_path,
                        texts_max_length=cfg['data']['max_text_len'],
                        features_max_length=cfg['data']['max_frame_len'])


train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

train_dataset.__getitem__(0)
print(len(train_dataloader))
print(len(val_dataloader))

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
vocab = tokenizer.vocab

model =  train(args, cfg, IPSLT, args.model_path, vocab, 
               train_dataloader, val_dataloader)