from torchvision.models.video import r3d_18 
import torch.nn as nn
from src.encoder import TransformerEncoder
import torch
from src.utils import create_trg_mask, create_mask
from src.model_utils import PositionalEncoding
import pytorch_lightning as pl
from einops import rearrange
from transformers import AutoTokenizer
import wandb


class IPSLT(pl.LightningModule):

    def __init__(self,
                cfg=None,
                args=None,
                text_vocab=None,
                K = 3, 
                KL_lambda=15,
                dim_feedforward_e=1024,
                dim_feedforward_d=1024,
                encoder_dim=512,  # Encoder dimension
                decoder_dim=512,  # Decoder dimension
                decoder_n_heads=4,
                encoder_n_heads=4,
                decoder_n_layers=3,
                encoder_n_layers=3,
                max_frame_len=300,
                dropout_e=0.1,
                dropout_d=0.1,
                activation_e="relu",
                activation_d="relu",
                base_learning_rate=1.0e-4,
                label_smoothing=0.1,
                emb_dim=512):
        super(IPSLT, self).__init__()

        self.cfg = cfg
        self.args = args
        self.text_vocab = text_vocab
        self.max_frame_len = max_frame_len
        
        if cfg['model'].get('K', None) != None:
            self.K = cfg['model']['K']
        else:
           self.K = K

        if cfg['model'].get('KL_lambda', None) != None:
            self.KL_lambda = cfg['model']['KL_lambda']
        else:
           self.KL_lambda = KL_lambda

        if cfg['model']['encoder'].get('n_layers', None) != None:
            self.encoder_n_layers = cfg['model']['encoder']['n_layers']
        else:
           self.encoder_n_layers = encoder_n_layers

        if cfg['model']['encoder'].get('n_head', None) != None:
            self.encoder_n_heads = cfg['model']['encoder']['n_head']
        else:
           self.encoder_n_heads = encoder_n_heads

        if cfg['model']['encoder'].get('dropout', None) != None:
            self.dropout_e = cfg['model']['encoder']['dropout']
        else:
           self.dropout_e = dropout_e
        
        if cfg['model']['encoder'].get('activation', None) != None:
            self.activation_e = cfg['model']['encoder']['activation']
        else:
           self.activation_e = activation_e
        
        if cfg['model']['encoder'].get('dim_feedforward', None) != None:
            self.dim_feedforward_e = cfg['model']['encoder']['dim_feedforward']
        else:
           self.dim_feedforward_e = dim_feedforward_e
        
        if cfg['model']['encoder'].get('encoder_dim', None) != None:
            self.encoder_dim = cfg['model']['encoder']['encoder_dim']
        else:
           self.encoder_dim = encoder_dim

        # ---- DECODER ---

        if cfg['model']['decoder'].get('n_layers', None) != None:
            self.decoder_n_layers = cfg['model']['decoder']['n_layers']
        else:
           self.decoder_n_layers = decoder_n_layers
        
        if cfg['model']['decoder'].get('n_head', None) != None:
            self.decoder_n_heads = cfg['model']['decoder']['n_head']
        else:
           self.decoder_n_heads = decoder_n_heads
        
        if cfg['model']['decoder'].get('decoder_dim', None) != None:
            self.decoder_dim = cfg['model']['decoder']['decoder_dim']
        else:
            self.decoder_dim = decoder_dim  
        
        if cfg['model']['decoder'].get('activation', None) != None:
            self.activation_d = cfg['model']['decoder']['activation']
        else:
           self.activation_d = activation_d

        if cfg['model']['decoder'].get('dropout', None) != None:
            self.dropout_d = cfg['model']['decoder']['dropout']
        else:
           self.dropout_d = dropout_d

        if cfg['model']['decoder'].get('dim_feedforward', None) != None:
            self.dim_feedforward_d = cfg['model']['decoder']['dim_feedforward']
        else:
            self.dim_feedforward_d = dim_feedforward_d

        if cfg['model']['embeddings'].get('embedding_dim', None) != None:
            self.emb_dim = cfg['model']['embeddings']['embedding_dim']
        else:
           self.emb_dim = emb_dim

        self.text_emb = nn.Linear(in_features=768, out_features=self.emb_dim)

        ## TRAINING
        if cfg['training'].get('base_learning_rate', None) != None:
            self.learning_rate = cfg['training']['base_learning_rate']
        else:
            self.learning_rate = base_learning_rate  

        if cfg['training'].get('label_smoothing', None) != None:
            self.label_smoothing = cfg['training']['label_smoothing']
        else:
           self.label_smoothing = label_smoothing

        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased") 

        self.training_step_outputs = []
        self.valid_step_outputs = []
        self.valid_losses = []
        self.train_losses = []

        # protoype inititialization
        encoder_layer1 = nn.TransformerEncoderLayer(self.encoder_dim, self.encoder_n_heads, self.dim_feedforward_e, self.dropout_e, self.activation_e, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(encoder_layer=encoder_layer1, num_layers=self.encoder_n_layers)

        decoder_layer1 = nn.TransformerDecoderLayer(self.decoder_dim, self.decoder_n_heads, self.dim_feedforward_d, self.dropout_d, self.activation_d, batch_first=True)
        self.decoder1 = nn.TransformerDecoder(decoder_layer=decoder_layer1, num_layers=self.decoder_n_layers)
        self.vocab_projection1 = nn.Linear(self.decoder_dim, self.tokenizer.vocab_size)

        # --- prototype refinement module ---
        self.encoder2 = TransformerEncoder(hidden_size = self.encoder_dim, ff_size= self.dim_feedforward_e, num_layers=self.encoder_n_layers, num_heads=self.encoder_n_heads, dropout=self.dropout_e)

        decoder_layer2 = nn.TransformerDecoderLayer(self.decoder_dim, self.decoder_n_heads, self.dim_feedforward_d, self.dropout_d, self.activation_d, batch_first=True)
        self.decoder2 = nn.TransformerDecoder(decoder_layer=decoder_layer2, num_layers=self.decoder_n_layers)

        self.pos_encoding = PositionalEncoding(self.decoder_dim)

        self.vocab_projection2 = nn.Linear(self.decoder_dim, self.tokenizer.vocab_size)

    
    def forward(self, text_embed, text_mask, features, features_mask):
        text_embed = self.pos_encoding(text_embed) 
        tgt_seq_len = text_embed.size(1)
        device = text_embed.device

        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device) == 1, diagonal=1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 1, float('-inf'))
        
        features = self.pos_encoding(features)

        decoder_outputs = []
        # --- PROTOTYPE INITIALIZATION MODULE ---
        enc_outs = self.encoder1(features, src_key_padding_mask=~features_mask)
        dec_outs = self.decoder1(
            tgt=text_embed,
            memory=enc_outs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~text_mask,              # (batch_size, tgt_seq_len)
            memory_key_padding_mask=~features_mask        # (batch_size, src_seq_len)
        )
        dec_outs = self.vocab_projection1(dec_outs)
        decoder_outputs.append(dec_outs)

        # --- PROTOTYPE REFINEMENT MODULE ---

        for i in range(self.K):
            enc_outs = self.encoder2(features, features_mask, enc_outs)

            dec_outs = self.decoder2(
                        tgt=text_embed,
                        memory=enc_outs,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=~text_mask,
                        memory_key_padding_mask=~features_mask
                        )
            dec_outs = self.vocab_projection2(dec_outs)
            decoder_outputs.append(dec_outs)

        return decoder_outputs


    def predict(self, features, features_mask, beam_size=5, max_len=52):
        batch_size = features.size(0)
        device = features.device

        # BOS token ID
        bos_token_id = self.text_vocab["[BOS]"]
        eos_token_id = self.text_vocab["[EOS]"]

        # Positional encoding
        features = self.pos_encoding(features)

        # Initial encoder output
        enc_outs = self.encoder1(features, src_key_padding_mask=~features_mask)
        for _ in range(self.K):
            enc_outs = self.encoder2(features, features_mask, enc_outs)

        # Each element in the batch will have `beam_size` candidates
        sequences = [[([bos_token_id], 0.0)] for _ in range(batch_size)]  # List of tuples (sequence, score)

        for step in range(max_len):
            all_candidates = []
            for b in range(batch_size):
                candidates = sequences[b]
                temp_candidates = []

                for seq, score in candidates:
                    if seq[-1] == eos_token_id:
                        temp_candidates.append((seq, score))
                        continue

                    # Prepare input
                    seq_tensor = torch.tensor(seq, device=device).unsqueeze(0)  # (1, seq_len)
                    text_embed = self.token_embedding(seq_tensor)  # Assuming you have token_embedding layer
                    text_embed = self.pos_encoding(text_embed)

                    tgt_seq_len = text_embed.size(1)
                    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device) == 1, diagonal=1)
                    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 1, float('-inf'))

                    dec_out = self.decoder2(
                        tgt=text_embed,
                        memory=enc_outs[b:b+1],  # select batch b
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=~features_mask[b:b+1]
                    )
                    logits = self.vocab_projection2(dec_out[:, -1, :])  # (1, vocab_size)
                    log_probs = torch.log_softmax(logits, dim=-1)  # (1, vocab_size)

                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

                    for i in range(beam_size):
                        new_seq = seq + [topk_indices[0, i].item()]
                        new_score = score + topk_log_probs[0, i].item()
                        temp_candidates.append((new_seq, new_score))

                # Keep top-k sequences
                ordered = sorted(temp_candidates, key=lambda tup: tup[1], reverse=True)
                sequences[b] = ordered[:beam_size]

        # Select best sequence for each item in batch
        final_sequences = []
        for b in range(batch_size):
            best_seq = max(sequences[b], key=lambda tup: tup[1])[0]
            final_sequences.append(best_seq)

        return final_sequences

    def share_step(self, tgt_ids, text_embed, text_mask, features, features_mask, split="train"):
        """
        One step of training/validation using BERT embeddings as input and TransformerDecoder outputs.
        Calculates cross-entropy loss for the first and last decoder outputs,
        and KL divergence loss between intermediate and final decoder outputs.
        """

        # Forward pass through the model
        decoder_outs = self(text_embed, text_mask, features, features_mask)

        # Split decoder outputs
        first_output = decoder_outs[0]         # Initial decoder output
        last_output = decoder_outs[-1]         # Final refined decoder output
        intermediate_outputs = decoder_outs[1:-1]  # All intermediate outputs
        tgt_ids = tgt_ids.to(first_output.device)

        # --- Cross Entropy Loss ---
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.text_vocab["[PAD]"])

        # Reshape decoder outputs and targets to compute CE loss
        ce_loss_first = ce_loss_fn(
            first_output.view(-1, first_output.size(-1)), 
            tgt_ids.view(-1)
        )
        ce_loss_last = ce_loss_fn(
            last_output.view(-1, last_output.size(-1)), 
            tgt_ids.view(-1)
        )


        # --- KL Divergence Loss ---
        # Measures how close intermediate outputs are to the final output
        kl_loss_fn = torch.nn.KLDivLoss(reduction="none")
        kl_div_loss_total = 0.0

        for inter_out in intermediate_outputs:
            kl_div = kl_loss_fn(
                torch.log_softmax(inter_out, dim=-1),              # log probabilities from intermediate decoder
                torch.softmax(last_output.detach(), dim=-1)        # target distribution from final decoder
            )
            mask = text_mask.unsqueeze(-1).expand_as(kl_div)  # [B, T, V]

            # Zero out the loss for padding positions
            kl_div = kl_div * mask  # [B, T, V]

            kl_div = kl_div.sum() / mask.sum()

            kl_div_loss_total += kl_div

        # Apply weighting factor to the KL divergence loss
        kl_div_loss_total = self.KL_lambda * kl_div_loss_total

        # --- Logging dictionary ---
        log_dict = {
            f"{split}/ce_loss_first": ce_loss_first.detach(),
            f"{split}/ce_loss_last": ce_loss_last.detach(),
            f"{split}/total_kl_div_loss": kl_div_loss_total.detach(),
            f"{split}/total_loss": total_loss.detach()
        }

        # Total loss is a combination of cross-entropy and KL losses
        total_loss = ce_loss_first + ce_loss_last + kl_div_loss_total
        log_dict[f"{split}/total_loss"] = total_loss.detach()

        wandb.log({
            f"{split}/ce_loss_first": ce_loss_first.detach(),
            f"{split}/ce_loss_last": ce_loss_last.detach(),
            f"{split}/total_kl_div_loss": kl_div_loss_total.detach(),
            f"{split}/total_loss": total_loss.detach()})

        generated_ids_last = torch.argmax(last_output, dim=-1)
        generated_text_last = self.tokenizer.batch_decode(generated_ids_last, skip_special_tokens=True)

        generated_ids_first = torch.argmax(first_output, dim=-1)
        generated_text_first = self.tokenizer.batch_decode(generated_ids_first, skip_special_tokens=True)

        wandb.log({f"{split}/sample_text_init": wandb.Html("<br>".join(generated_text_first))})
        wandb.log({"f{split}/sample_text_last": wandb.Html("<br>".join(generated_text_last))})


        return total_loss, log_dict

    
    def get_inputs(self, batch):
        text_embed = batch["text_embeddings"]
        text = batch["text"]
        encoding = self.tokenizer(
            text,
            padding="max_length",     
            max_length=text_embed.shape[1],
            truncation=True,
            return_tensors="pt"
        )

        tgt_ids = encoding["input_ids"] 

        features = batch["features"]
        features_lengths = batch["features_length"]

        features_mask = create_mask(features_lengths, self.max_frame_len, device=self.device)

        text_mask = (text_embed != self.text_vocab["[PAD]"])
        text_mask = text_mask[:, :, 0] # BERT embeddings

        # Map tensor to device
        text_embed, features = map(lambda tensor: tensor.to(self.device), [text_embed, features])
        text_mask, features_mask = map(lambda tensor: tensor.to(self.device), [text_mask, features_mask])

        return (tgt_ids, text_embed, text_mask), (features, features_mask)
    
    def training_step(self, batch, batch_idx):

        (tgt_ids, text_embed, text_mask), (features, features_mask) = self.get_inputs(batch)

        total_loss, log_dict = self.share_step(tgt_ids, text_embed, text_mask, features, features_mask, split="train")
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        self.training_step_outputs.append(total_loss)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # Log the learning rate
        self.log('lr', lr, prog_bar=True, logger=True, sync_dist=True)
        
        return total_loss

    def on_train_epoch_end(self):
        # Calculate and store the mean loss for the epoch
        epoch_loss = torch.stack([loss for loss in self.training_step_outputs]).mean()
        self.train_losses.append(epoch_loss.item())   

    def validation_step(self, batch, batch_idx):
        (tgt_ids, text_embed, text_mask), (features, features_mask) = self.get_inputs(batch)
        total_loss, log_dict = self.share_step(tgt_ids, text_embed, text_mask, features, features_mask, split="valid")
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        self.valid_step_outputs.append(total_loss)

        return total_loss
    def on_validation_epoch_end(self):
        # Calculate and store the mean loss for the epoch
        epoch_loss = torch.stack([loss for loss in self.valid_step_outputs]).mean()
        self.valid_losses.append(epoch_loss.item())

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=lr, 
            weight_decay=0.0005
        )

        # ReduceLROnPlateau Scheduler (Train Loss'a Göre)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Minimum train loss izlenir
            factor=0.9,           # LR 0.5 ile çarpılır
            patience=40,          # 40 epoch boyunca iyileşme olmazsa LR azaltılır
            verbose=True,         # LR azaldığında log basılır
            min_lr=1e-12          # Minimum öğrenme oranı
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid/total_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }