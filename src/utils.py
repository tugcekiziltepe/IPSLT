import random
import torch
import numpy as np

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