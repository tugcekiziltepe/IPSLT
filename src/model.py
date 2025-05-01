from torchvision.models.video import r3d_18 
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder
import torch
from utils import create_trg_mask

def make_i3d(device):
    i3d = r3d_18(pretrained=True).to(device)
    i3d.eval()
    return i3d 

from sacrebleu.metrics import BLEU
from rouge import Rouge


class InitializationModule(nn.Module):
    def __init__(self,
                hidden_size = 400,
                ff_size = 2048,
                num_layers = 3,
                num_heads = 8,
                dropout= 0.1):
        super(InitializationModule, self).__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()

    def forward(self, video, translation,src_mask, tgt_mask):
    
        encoded = self.encoder(video, src_mask, initial = True)
        decoded = self.decoder(translation, encoded, src_mask=src_mask, trg_mask=tgt_mask)

        return encoded, decoded

class IterativePrototypeRefinement(nn.Module):
    def __init__(self,
                hidden_size = 400,
                ff_size = 2048,
                num_layers = 3,
                num_heads = 8,
                dropout= 0.1,
                K = 3):
        super(IterativePrototypeRefinement, self).__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.K = 3

    def forward(self, video, encoded_mem, translation,src_mask, tgt_mask):
        
        decoder_outputs = []
        for i in range(self.K):
            encoded = self.encoder(video, src_mask, encoded_mem)
            decoded = self.decoder(video, encoded, src_mask=src_mask, trg_mask=tgt_mask)
            decoder_outputs.append(decoded[0].detach().cpu())
        return decoder_outputs


class IPSLT(nn.Module):

    def __init__(self,
                device,
                hidden_size = 400,
                ff_size = 2048,
                num_layers = 3,
                num_heads = 8,
                dropout= 0.1, K = 3):
        super(IPSLT, self).__init__()
        self.initalization_module = InitializationModule()
        self.iterative_prototype_refinement = IterativePrototypeRefinement()
        # Parameters
        vocab_size = 1000  # Example vocabulary size
        embedding_dim = 400  # Number of features per embedding

        # Creating the embedding layer
        self.embedding_layer = nn.Embedding(31102, embedding_dim)
        self.i3d = make_i3d(device)
        self.device = device
    
    

    def forward(self, batch_video_clips, translation):
        # translation shape: (batch_size x 1 x 400)
        embeddings = self.embedding_layer(translation.to(self.device)) # batch_size x 1 x seq_length x feature_size
        batch_video_clips = batch_video_clips.squeeze(1).permute(0, 2, 1, 3, 4).to(self.device).float() 
        print(batch_video_clips.shape)
        features = self.i3d(batch_video_clips) # batch_size x feature_size
        # # Concatenate the original tensor with the zeros tensor
        sgn_mask = (features != torch.zeros(400).to(self.device))[..., 0].to(self.device)#.unsqueeze(1) # # batch_size x batch_size x 1 x 400
        trg_mask = create_trg_mask(translation, 0).to(self.device) # batch_size x 400 x 400
        print(trg_mask.shape)
        # decoder_outputs_new = []
        encoded, decoder_output = self.initalization_module(features, embeddings, sgn_mask, trg_mask)
        # decoder_outputs_new.append(decoder_output[0].cpu())
        # decoder_outputs = self.iterative_prototype_refinement(features_video_clips, encoded, embeddings, sgn_mask, trg_mask.repeat(8, 1, 1))
        # decoder_outputs_new.extend(decoder_outputs)
        # features_video_clips = features_video_clips.detach().cpu()
        # del features_video_clips
        # return decoder_outputs_new