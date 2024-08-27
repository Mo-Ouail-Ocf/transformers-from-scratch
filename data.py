from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
import torch
from torch import Tensor
from config import Config

from pathlib import Path

from torch.utils.data import random_split , DataLoader , Dataset

def get_all_sentences(ds, lang:str):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config:Config,dataset,lang:str)->Tokenizer:
    tokenizer_path = Path(config.tokenizer_file.format(lang))

    if tokenizer_path.exists():
       return Tokenizer.from_file(str(tokenizer_path))

    # Build the tokenizer 
    tokenzier = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenzier.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=[
            "[UNK]","[PAD]","[EOS]","[SOS]"
        ] ,
        min_frequency=2,
        show_progress=True, 
    )
    tokenzier.train_from_iterator(get_all_sentences(dataset,lang),
                                  trainer=trainer)

    tokenzier.save(str(tokenizer_path))
    return tokenzier


 
def get_dataset(config:Config):
    dataset = load_dataset(config.datasource,f'{config.lang_tgt}-{config.lang_src}',split='train')
    
    # build tokenizers
    src_tokenizer = get_or_build_tokenizer(config,dataset,lang=config.lang_src)
    tgt_tokenizer = get_or_build_tokenizer(config,dataset,lang=config.lang_tgt)

    train_ds_size = int(0.9 * len(dataset))
    valid_ds_size =len(dataset) - train_ds_size

    train_ds_raw , valid_ds_raw = random_split(dataset,lengths=[train_ds_size,valid_ds_size])

    train_ds  = EnglishArabicDataset(
        train_ds_raw,src_tokenizer,tgt_tokenizer,config.lang_src,config.lang_tgt,config.seq_len
    )
    valid_ds  = EnglishArabicDataset(
        valid_ds_raw,src_tokenizer,tgt_tokenizer,config.lang_src,config.lang_tgt,config.seq_len
    )

    train_dl =DataLoader(train_ds,batch_size=config.batch_size,shuffle=True)
    valid_dl =DataLoader(valid_ds,batch_size=config.batch_size,shuffle=True)

    return train_dl,valid_dl,src_tokenizer,tgt_tokenizer


class EnglishArabicDataset(Dataset):
    def __init__(self,dataset,src_tokenizer:Tokenizer,tgt_tokenizer:Tokenizer,
                 src_lang:str,tgt_lang:str,seq_len:int,device='cuda'):
        super().__init__()
        self.dataset = dataset

        self.src_tokenizer=src_tokenizer
        self.src_lang=src_lang

        self.tgt_tokenizer=tgt_tokenizer
        self.tgt_lang=tgt_lang

        self.seq_len=seq_len

        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")],dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")],dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")],dtype=torch.int64)

        self.device = device

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) :
        src_target_pair = self.dataset[index]['translation']

        src_sentence , tgt_sentence = src_target_pair[self.src_lang],\
                                    src_target_pair[self.tgt_lang]    

        encoded_src_sentence = self.src_tokenizer.encode(src_sentence).ids
        encoded_tgt_sentence = self.tgt_tokenizer.encode(tgt_sentence).ids

        nb_pads_src = self.seq_len - len(encoded_src_sentence) -2
        nb_pads_tgt = self.seq_len - len(encoded_tgt_sentence) -1

        encoder_input = torch.cat([
            self.sos_token ,
            torch.tensor(encoded_src_sentence,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*nb_pads_src)
        ])

        # Add SOS token to decoder input : shifted input according to paper
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(encoded_tgt_sentence,dtype=torch.int64) ,
            torch.tensor([self.pad_token]*nb_pads_tgt)
        ])

        # Add EOS to target to indicate end of sentence
        target = torch.cat([
            torch.tensor(encoded_tgt_sentence,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*nb_pads_tgt,dtype=torch.int64)
        ])


        assert encoder_input.shape[0]==self.seq_len
        assert decoder_input.shape[0]==self.seq_len
        assert target.shape[0]==self.seq_len

        return {
            'encoder_input':encoder_input.to(self.device) ,# (seq_len,)
            'encoder_mask':(encoder_input != self.pad_token).int().\
                        unsqueeze(0).unsqueeze(0).to(self.device) , # (1,1,seq_len) to broadcast to (heads,seq_len,seq_len)
            'decoder_input':decoder_input, # (seq_len,)
            'decoder_mask': (self._get_causal_mask(decoder_input.shape[0]) & (decoder_input != self.pad_token).int().\
                        unsqueeze(0)).to(self.device),# (1,seq_len,seq_len) & (1,seq_len)
            # (1, 1, seq_len) ->when broadcasting and batching (batch_size,nb_heads,seq_len,seq_len)
            'target':target.to(self.device), # (seq_len,)
        }

    def _get_causal_mask(self,seq_len:int)->Tensor:
        mask = torch.triu(torch.ones(1,seq_len,seq_len),diagonal=1).type(torch.int)
        return mask == 0 # (1,seq_len,seq_len)



if __name__=="__main__":
    config=Config()
    train_dl , valid_dl ,_,_ = get_dataset(config=config)
    data = next(iter(valid_dl))
    encoder_inputs = data['encoder_input']
