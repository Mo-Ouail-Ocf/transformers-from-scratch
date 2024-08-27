import torch
import torch.nn as nn
from torch import Tensor
import math
import typing as tt 
from dataclasses import dataclass
from config import Config

class InputEmbedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model= d_model
        self.embed=nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=d_model)
        
    def forward(self,x):
        # x : [batch_size,seq_len] , out : [batch_size,seq_len,embed_dim]
        return self.embed(x) * math.sqrt(self.d_model)
    # output : []


class  PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.len_seq = seq_len
        self.d_model = d_model
        self.pe = self._create_pe()
        self.register_buffer('pe',self.pe) # (seq_len , d_model)
        self.dropout = nn.Dropout(dropout)

    def _create_pe(self)->Tensor:
        pe = torch.zeros(self.len_seq,self.d_model)
        pos = torch.arange(0,self.len_seq,dtype=torch.float).unsqueeze(-1) # ---> (seq_len,1) - > (seq_len,d_model) after broadcasting
        div = torch.exp( -math.log(10_000)*torch.arange(0,self.d_model,2).float() / self.d_model ) # -->(d,model,) - > (seq_len,d_model) after broadcasting
        pe[:,0::2] = torch.sin(pos*div)      
        pe[:,1::2] = torch.cos(pos*div)

        return pe.unsqueeze(0)

    def forward(self,x:Tensor)->Tensor:
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False) # (1, x.shape[1], d_model)
        return self.dropout(x)
    

class LayerNorm(nn.Module):
    def __init__(self,eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self,x:Tensor)->Tensor:
        # input : (batch_size , seq_lenght , d_model)
        mean = x.mean(dim=-1,keepdim=True) # (batch_size , seq_lenght , 1)
        std = x.std(dim=-1,keepdim=True)
        std_x = (x-mean)/(std+self.eps)
        return self.alpha*std_x+self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self,drop:float,d_ff:int=2048,d_model=512):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(d_ff,d_model),
        )
    
    def forward(self,x)->Tensor:
        return self.ff(x)
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,dropout:float,seq_len:int,d_model:int,h:int):
        super().__init__()
        self.drop =dropout
        self.d_model = d_model
        self.seq_len = seq_len
        self.h = h # nb heads
        assert d_model % h == 0 , "d_model must be divisible by  h"
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)

        self.W_o = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout)

    def _get_heads(self,T:Tensor)->Tensor:
        # we return new shape [ Batch size , Nb heads , Seq Len , d_k ]
        return T.view(T.shape[0],T.shape[1],self.h,self.d_k).transpose(1,2)

    @staticmethod
    def calc_attention(Query_h:Tensor,Key_h:Tensor,Value_h:Tensor,dropout:nn.Dropout,Mask:Tensor|None=None)->tt.Tuple[Tensor,Tensor]:
        logit_att_scores = Query_h @ Key_h.transpose(-2,-1) # (batch_size,h,seq_len,seq_len)
        # mask the values
        if Mask is not None:
            logit_att_scores.masked_fill_(Mask==0,-1e9)
        attention_scores = logit_att_scores.softmax(dim=-1)  # (batch_size,h,seq_len,seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ Value_h , attention_scores
        

    def forward(self,Q:Tensor,K:Tensor,V:Tensor,Mask:Tensor|None=None)->Tensor:
        # All of shapes (seq_len,d_model)
        Query =  self.W_q(Q)
        Key =  self.W_k(K)
        Value =  self.W_v(V)

        # Divide query ,key & value into d_k heads
        Query_h = self._get_heads(Query)
        Key_h = self._get_heads(Key)
        Value_h = self._get_heads(Value)

        x , attention_scores = MultiHeadAttentionBlock.calc_attention(Query_h,Key_h,Value_h,self.dropout,Mask)
        # concat :
        # (batch_size,h,seq_len,d_k) -> (batch_size,seq_len,h,d_k) -> (batch_size,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.d_model)

        return self.W_o(x) # (batch_size,seq_len,d_model)
    

class ResidualConnection(nn.Module):
    # Add & Norm connection
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout =nn.Dropout(dropout)
        self.norm_layer = LayerNorm()

    def forward(self,x:Tensor,sub_layer:nn.Module)->Tensor:
        # Norm 
        norm_x = self.norm_layer(x)
        # Sub Layer output 
        out_layer = sub_layer(norm_x)
        # Dropout
        dropped_output = self.dropout(out_layer)
        # Residual connection
        out = x+dropped_output
        return out


class EncoderBlock(nn.Module):
    def __init__(self,dropout:float,
                 self_attention_block:MultiHeadAttentionBlock,
                 fc_block:FeedForwardBlock):
        super().__init__()
        self.self_att_block = self_attention_block
        self.fc_block = fc_block
        self.residual_connects = nn.ModuleList(
            [
                ResidualConnection(dropout)
                for _ in range(2)
            ]
        )

    def forward(self,x:Tensor,src_mask:Tensor|None=None)->Tensor:
        self_mha_out = self.residual_connects[0](
            x, lambda x: self.self_att_block(x,x,x,src_mask)
        )
        fc_out = self.residual_connects[1](
            self_mha_out , self.fc_block
        )
        return fc_out

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList) :
        super().__init__() 
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self,x:Tensor,mask:Tensor)->Tensor:
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
     
fc_block = FeedForwardBlock(drop=0.5)
mha_block = MultiHeadAttentionBlock(dropout=0.5,seq_len=120,d_model=512,h=8)

x = torch.zeros(1,120,512)

encdr = EncoderBlock(0.5,mha_block,fc_block)

class DecoderBlock(nn.Module):
    def __init__(self,
                 dropout:float,
                masked_mha:MultiHeadAttentionBlock,
                cross_mha:MultiHeadAttentionBlock,
                fc_block:FeedForwardBlock) -> None:
        super().__init__()
        self.masked_mha = masked_mha
        self.cross_mha= cross_mha
        self.fc_block = fc_block
        self.residual_connections = nn.ModuleList(
            [
                ResidualConnection(dropout)
                for _ in range(3)
            ]
        )
    
    def forward(self,x:Tensor,encoder_out:Tensor,src_mask:Tensor,tgt_mask:Tensor)->Tensor:
        masked_mha_out = self.residual_connections[0](
            x , lambda x : self.masked_mha(x,x,x,tgt_mask)
        ) 
        cross_att_out = self.residual_connections[1](
            masked_mha_out , lambda masked_mha_out : self.cross_mha(masked_mha_out,
                                                                    encoder_out,encoder_out,src_mask)
        )
        fc_out = self.residual_connections[2](
            cross_att_out , self.fc_block
        )
        return fc_out
    
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList) :
        super().__init__() 
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self,x:Tensor,encoder_out:Tensor,src_mask:Tensor,tgt_mask)->Tensor:
        for layer in self.layers:
            x = layer(x,encoder_out,src_mask,tgt_mask)
        return self.norm(x)

    
class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x:Tensor)->Tensor:
        return torch.log_softmax(self.proj(x),dim=-1)
    # Output is (batch size , seq len , vocab size)


class Transformer(nn.Module):
    def __init__(self,src_embed:InputEmbedding,tgt_embed:InputEmbedding,
                 src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,
                 encoder:Encoder,decoder:Decoder,
                 project_layer:ProjectionLayer):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.project_layer = project_layer

    def encode(self,input_seq,source_mask:Tensor)->Tensor:
        x = self.src_embed(input_seq)
        x = self.src_pos(x)
        out_encd = self.encoder(x,source_mask)
        return out_encd
    
    # x:Tensor,encoder_out:Tensor,src_mask:Tensor,tgt_mask
    def decode(self,tgt_seq,encoder_out:Tensor,src_mask:Tensor,tgt_mask:Tensor)->Tensor:
        x = self.tgt_pos(self.tgt_embed(tgt_seq))
        out_put_dec = self.decoder(x,encoder_out,src_mask,tgt_mask)
        return out_put_dec
    
    def project(self,x:Tensor)->Tensor:
        return self.project_layer(x)

@dataclass
class BuildTransformer():
    # inputs
    src_vocab_size:int
    src_seq_len:int

    tgt_vocab_size:int
    tgt_seq_len:int

    # According to original paper
    d_model:int=512

    N:int=6 # nb encd , decd blocks to use

    h:int=8 # nb heads for each end ,decd block

    dropout:float=0.1

    d_ff:int=2048


def build_encoder(hyperparams:BuildTransformer):
    # encoder_block : self mha block + fc block
    self_mha_block = MultiHeadAttentionBlock(
        dropout=hyperparams.dropout,
        seq_len=hyperparams.src_seq_len,
        d_model=hyperparams.d_model,
        h=hyperparams.h
    )
    fc_block = FeedForwardBlock(
        drop=hyperparams.dropout,
        d_ff=hyperparams.d_ff,
        d_model=hyperparams.d_model,
    )

    encoder_block = EncoderBlock(dropout=hyperparams.dropout,
                                 self_attention_block=self_mha_block,fc_block=fc_block)
    
    encoder_blocks_list = nn.ModuleList()

    for _ in range(hyperparams.N):
        encoder_blocks_list.append(encoder_block)

    encoder = Encoder(encoder_blocks_list)

    return encoder

def build_decoder(hyperparams:BuildTransformer):
    # encoder_block : self mha block + cross mha block + fc block
    self_mha_block = MultiHeadAttentionBlock(
        dropout=hyperparams.dropout,
        seq_len=hyperparams.tgt_seq_len,
        d_model=hyperparams.d_model,
        h=hyperparams.h
    )
    cross_mha_block = MultiHeadAttentionBlock(
        dropout=hyperparams.dropout,
        seq_len=hyperparams.tgt_seq_len,
        d_model=hyperparams.d_model,
        h=hyperparams.h
    )
    fc_block = FeedForwardBlock(
        drop=hyperparams.dropout,
        d_ff=hyperparams.d_ff,
        d_model=hyperparams.d_model,
    )

    decoder_block = DecoderBlock(dropout=hyperparams.dropout,
                                 cross_mha=cross_mha_block,
                                 masked_mha=self_mha_block,
                                 fc_block=fc_block)
    
    decoder_blocks_list = nn.ModuleList()

    for _ in range(hyperparams.N):
        decoder_blocks_list.append(decoder_block)

    decoder = Decoder(layers=decoder_block)

    return decoder



def build_transformer(
        hyperparams:BuildTransformer
    )->Transformer:
    # create input layers to encoder / decoder
    src_embed,tgt_embed = InputEmbedding(hyperparams.src_vocab_size,hyperparams.d_model),\
                            InputEmbedding(hyperparams.tgt_vocab_size,hyperparams.d_model)
    
    src_pos_encd ,tgt_pos_encd = PositionalEncoding(d_model=hyperparams.d_model,seq_len=hyperparams.src_seq_len),\
                                PositionalEncoding(d_model=hyperparams.d_model,seq_len=hyperparams.tgt_seq_len)
    
    # Encoder , Decoder
    encoder , decoder = build_encoder(hyperparams) , build_decoder(hyperparams)
    # Output of transformer:
    project_layer = ProjectionLayer(hyperparams.d_model,hyperparams.tgt_vocab_size)

    transformer = Transformer(
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        src_pos=src_pos_encd,
        tgt_pos=tgt_pos_encd,
        encoder=encoder,
        decoder=decoder,
        project_layer=project_layer,
    )

    # init params

    for param in transformer.parameters():
        if param.dim()>1:
            nn.init.xavier_uniform_(param)

    return transformer


def get_model(config:Config,src_vocab_size:int,tgt_vocab_size:int,device='cuda')->tt.Tuple[Transformer,torch.optim.Adam]:
    seq_len = config.seq_len
    hyperparams = BuildTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        tgt_seq_len=seq_len,
        src_seq_len=seq_len,
    )
    transformer = Transformer(hyperparams).to(device)
    optimizer = torch.optim.Adam(
        params=transformer.parameters(),
        lr=config.lr
    )
    return transformer , optimizer
