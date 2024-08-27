import torch.nn as nn
import torch

from ignite.engine import Engine

from data import get_dataset
from model import get_model
from config import Config

from utils import attach_ignite,resume_from_checkpoint

DEVICE = 'cuda'

config = Config()

train_dl , valid_dl , src_tokenizer , tgt_tokenizer = get_dataset(config)

transformer,optimizer = get_model(config,src_tokenizer.get_vocab_size,
                                    tgt_tokenizer.get_vocab_size)


loss_fn = nn.CrossEntropyLoss(
    ignore_index=src_tokenizer.token_to_id('[PAD]'),
    label_smoothing=0.1
    ).to(DEVICE)


def train_step(engine:Engine,batch):
    transformer.train()

    # Unpack batch
    encoder_input = batch['encoder_input'] # (batch_size, seq_len)
    encoder_mask = batch['encoder_mask']   # (batch_size,1,1,seq_len)
    decoder_input = batch['decoder_input']
    decoder_mask = batch['decoder_mask']
    targets = batch['target']               # (batch_size , seq_len)

    # Calc targets
    encoder_output = transformer.encode(encoder_input,encoder_mask) #(batch_size,seq_len,d_model)
    decoder_output = transformer.decode(decoder_input,encoder_output,encoder_mask,decoder_mask) #(batch_size,seq_len,d_model)
    predicted_logits = transformer.project(decoder_output) # #(batch_size,seq_len,vocab_size) 

    # adjust dimensions to match in loss
    predicted_logits = predicted_logits.view(-1,tgt_tokenizer.get_vocab_size()) #(-1,vocab_size)
    targets = targets.view(-1) # (-1)

    # Loss & optimize
    loss = loss_fn(predicted_logits,targets)
    transformer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss' : loss.item()
    }

@torch.no_grad()
def valid_step(engine:Engine,batch):
    transformer.eval()

    # Unpack batch
    encoder_input = batch['encoder_input'] # (batch_size, seq_len)
    encoder_mask = batch['encoder_mask']   # (batch_size,1,1,seq_len)
    decoder_input = batch['decoder_input']
    decoder_mask = batch['decoder_mask']
    targets = batch['target']               # (batch_size , seq_len)

    # Calc targets
    encoder_output = transformer.encode(encoder_input,encoder_mask) #(batch_size,seq_len,d_model)
    decoder_output = transformer.decode(decoder_input,encoder_output,encoder_mask,decoder_mask) #(batch_size,seq_len,d_model)
    predicted_logits = transformer.project(decoder_output) # #(batch_size,seq_len,vocab_size) 

    # adjust dimensions to match in loss
    predicted_logits = predicted_logits.view(-1,tgt_tokenizer.get_vocab_size()) #(-1,vocab_size)
    targets = targets.view(-1) # (-1)

    return predicted_logits,targets

# Resume training from checkpoint if available

trainer = Engine(train_step)
evaluator = Engine(valid_step)

attach_ignite(trainer,evaluator,transformer,optimizer,loss_fn,valid_dl)



if __name__=="__main__":
    # resume training in case there was a break
    resume_from_checkpoint(trainer, transformer, optimizer, config)
    trainer.run(train_dl, max_epochs=config.num_epochs)