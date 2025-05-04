import os
import shutil
from tqdm import tqdm

root_dir = "/Users/tugcekiziltepe/Desktop/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
embeddings_dir = "dino_embeddings"

# Yeni klasörleri oluştur
for split in ['train', 'test', 'dev']:
    split_output_dir = os.path.join(embeddings_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    split_input_dir = os.path.join(root_dir, split)

    if not os.path.exists(split_input_dir):
        continue

    video_names = [
        name for name in os.listdir(split_input_dir)
        if os.path.isdir(os.path.join(split_input_dir, name))
    ]

    for video_name in tqdm(video_names, desc=f"Moving embeddings for {split}"):
        source_file = os.path.join(embeddings_dir, f"{video_name}.npy")
        
        # Eğer kaynak dosya varsa hedef dizine taşı
        if os.path.exists(source_file):
            target_file = os.path.join(split_output_dir, f"{video_name}.npy")
            shutil.copy(source_file, target_file)

print("✅ Embeddings have been successfully moved to separate train/test/dev directories.")
