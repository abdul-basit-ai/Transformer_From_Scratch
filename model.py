#My
import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vovab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.embedding.embedding_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len:int, dropout:int)-> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #Create a matrix of shape (seq_len, d_model) to hold the positional encodings
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) #Implemented the eqn in log for ease of computation
        #Apply the sine function to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        #Apply the cosine function to odd indices in the array; 2i+1             
        pe[:, 1::2] = torch.cos(position * div_term)    
        
        pe = pe.unsqueeze(0)  # Shape (1, seq_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNorm(nn.Module):
    def __init__(self,feature:int, eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(feature))#Multiplied
        self.bias = nn.Parameter(torch.zeros(feature))#Addeed
        
    def forward(self, x):
        mean = x.mean (dim=-1, keepdim=True)
        std = x.std (dim=-1, keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias
        
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float)->None:
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)#W1 ansd B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model) #W2 and B2
        
        
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__ (self, d_model:int, h:int, dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model,bias=False) #Wq
        self.w_k = nn.Linear(d_model, d_model,bias=False) #Wk
        self.w_v = nn.Linear(d_model, d_model,bias=False) #Wv
        
        self.w_o = nn.Linear(d_model, d_model,bias=False) #Wo
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot-product attention
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e-9)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores  # (batch_size, h, seq_len, d_k), (batch_size, h, seq_len, seq_len)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
    
    def forward(self, query, key, value, mask):
        query = self.w_q(query) # (batch_size, seq_len, d_model)
        key = self.w_k(key)     # (batch_size, seq_len, d_model)
        value = self.w_v(value) # (batch_size, seq_len, d_model)
        
        query = query.view(query.shape(0), query.shape[1], self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)
        key = key.view(key.shape(0), key.shape[1], self.h, self.d_k).transpose(1, 2)         # (batch_size, h, seq_len, d_k)
        value = value.view(value.shape(0), value.shape[1], self.h, self.d_k).transpose(1, 2)    # (batch_size, h, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        # Concatenate heads and project back to d_model
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape(0), -1, self.h * self.d_k)
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)  # (batch_size, seq_len, d_model) 

class ResidualConnection(nn.Module):
    def __init__(self,features:int, dropout:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder_Block(nn.Module):
    def __init__(self, features:int,self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float)-> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, features:int,layers: nn.ModuleList)-> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder_Block(nn.Module):
    def __init__(self, features:int,self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float)-> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
        
class Decoder(nn.Module):
    def __init__ (self,features:int, layers: nn.ModuleList) -> None: 
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size)->None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        #Batch, seq_len, d_model--> Batch, seq_len, vocab_size
        return self.proj(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        # batch, seq_len --> batch, seq_len, d_model
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        # same here, batch, seq_len --> batch, seq_len, d_model
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask) 
    
    def project(self, x):
        return self.projection_layer(x)
    
def Build(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model:int=512, N:int=6, h:int=8, dropout:float=0.1, d_ff:int=2048)->Transformer:
    #Create the embedding and positional encoding layers for both source and target
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    #Create the encoder and decoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = Encoder_Block(d_model,encoder_self_attention, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        cross_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = Decoder_Block(d_model,decoder_self_attention, cross_attention, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block) 
        
    #Create Encoder and Decoder     
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    #Create the projection layer      
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    #parameters initialize
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
