from data import PHEONIX14T
from torchvision import transforms
from transformers import BertTokenizer
from config import config
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import torch.optim as optim
from model import IPSLT
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    train_corpus_path = r"D:\phoenix-2014-T.v3\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\annotations\manual\PHOENIX-2014-T.train.corpus.csv"
    dev_corpus_path = r"D:\phoenix-2014-T.v3\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\annotations\manual\PHOENIX-2014-T.dev.corpus.csv"
    test_corpus_path = r"D:\phoenix-2014-T.v3\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\annotations\manual\PHOENIX-2014-T.test.corpus.csv"

    train_videos_path = r"D:\phoenix-2014-T.v3\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\features\fullFrame-210x260px\train"
    dev_videos_path = r"D:\phoenix-2014-T.v3\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\features\fullFrame-210x260px\dev"
    test_videos_path = r"D:\phoenix-2014-T.v3\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\features\fullFrame-210x260px\test"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    data_transform = transforms.Compose([
                                    transforms.ToTensor()
                                    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-dbmdz-uncased')

    # Get the size of the tokenizer's vocabulary
    vocab_size = len(tokenizer.vocab)
    print("Vocabulary Size:", vocab_size)

    train_data = PHEONIX14T(train_corpus_path, train_videos_path, data_transform, config["pad_feature_size"], config["max_sent_length"], tokenizer)
    dev_data = PHEONIX14T(dev_corpus_path, dev_videos_path, data_transform, config["pad_feature_size"],  config["max_sent_length"], tokenizer)
    dev_data = PHEONIX14T(dev_corpus_path, dev_videos_path, data_transform, config["pad_feature_size"],  config["max_sent_length"], tokenizer)
    test_data = PHEONIX14T(test_corpus_path, test_videos_path, data_transform, config["pad_feature_size"], config["max_sent_length"], tokenizer)

    print(train_data.__getitem__(1)[2].shape)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=2,
                              shuffle=True,
                              worker_init_fn=seed_worker)

    dev_dataloader = DataLoader(dataset=dev_data, 
                                batch_size=2, 
                                shuffle=False,
                                worker_init_fn=seed_worker) 

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=2, 
                                shuffle=False,
                                worker_init_fn=seed_worker) 
    
    print(f"Train dataloader lenght: {len(train_dataloader)}, Dev dataloader length {len(dev_dataloader)}, Test datalodar length {len(test_dataloader)}  ")


    # Initialize TensorBoard writer
    writer = SummaryWriter('runs\experiment_name')
    model = IPSLT(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    epochs = 50

    train_losses = []
    dev_losses = []
    best_val_loss = float('inf')
    for epoch in range(7, epochs):
        model.train()  # Training mode
        total_train_loss = 0
        for batch, (imgs, gloss, translation) in enumerate(train_dataloader):
            optimizer.zero_grad()
            decoder_outputs = model(imgs, translation)  