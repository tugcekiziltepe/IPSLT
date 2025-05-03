import random
import torch
import numpy as np
import yaml

def set_seet(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)

    g = torch.Generator()
    g.manual_seed(0)

def create_look_ahead_mask(size):
    # Create a matrix where the entries above the diagonal are True (masked)
    mask = torch.triu(torch.ones((size, size)), diagonal=1).bool()
    return mask

def create_trg_mask(trg, pad_token_id):
    # trg: [batch_size, trg_len]
    # pad_token_id: the token ID used for padding (e.g., BERT's [PAD] token ID)

    # Create a padding mask for ignoring pad tokens
    pad_mask = (trg == pad_token_id).unsqueeze(1)  # Shape: [batch_size, 1, trg_len]

    # Create the look-ahead mask
    trg_len = trg.size(1)
    look_ahead_mask = create_look_ahead_mask(trg_len)  # Shape: [trg_len, trg_len]
    look_ahead_mask = look_ahead_mask.to(trg.device).expand(trg.size(0), trg_len, trg_len)  # [batch_size, trg_len, trg_len]

    # Combine the masks
    trg_mask = pad_mask | look_ahead_mask  # Shape: [batch_size, trg_len, trg_len]
    return trg_mask

def load_config(path="train.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def pad_or_truncate_frames(a, max_len):
    """
    Pads or truncates the input array to match the given max length.
    
    - If the sequence has fewer frames than max_len, pad with value 2.
    - If the sequence has more frames than max_len, randomly remove frames.
    
    Parameters:
        a (np.ndarray): Input array with shape (num_frames, ...).
        max_len (int): Desired sequence length.
    
    Returns:
        np.ndarray: Processed array with shape (max_len, ...).
    """
    num_frames = a.shape[0]
    
    if num_frames == max_len:
        return a  # No change needed
    
    elif num_frames > max_len:
        # Randomly select max_len indices to keep
        indices = np.sort(np.random.choice(num_frames, max_len, replace=False))
        return a[indices]
    
    else:
        # Pad with value 2
        pad_shape = (max_len - num_frames, *a.shape[1:])
        return np.concatenate((a, np.full(pad_shape, 2)), axis=0)


def create_mask(seq_lengths, max_len, device="cpu"):
    # mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device).clone().detach()[:, None]
    mask = torch.arange(max_len, device=device)[None, :] < seq_lengths.clone().detach()[:, None]
    return mask.bool()
