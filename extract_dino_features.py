

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Directories
root_dir = "/Users/tugcekiziltepe/Desktop/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
output_dir = "dino_embeddings"
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained DINO ViT model from Hugging Face
model_name = "facebook/dino-vitb16"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

def get_dino_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding.squeeze()

def process_video(video_path):
    frame_files = sorted([
        os.path.join(video_path, f)
        for f in os.listdir(video_path)
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ])
    
    embeddings = [get_dino_embedding(frame) for frame in frame_files]
    embeddings = np.stack(embeddings)
    
    return embeddings

# Main loop
for split in ['train', 'test', 'dev']:
    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        continue

    video_names = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

    for video_name in tqdm(video_names, desc=f"Processing {split}"):
        video_dir = os.path.join(split_dir, video_name)
        
        embeddings = process_video(video_dir)
        
        npy_path = os.path.join(output_dir, f"{video_name}.npy")
        np.save(npy_path, embeddings)

print("✅ DINO Embeddings have been successfully created and saved at:", output_dir)
