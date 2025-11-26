import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self , d_model:int , vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model)

    def forward(self , x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self , d_model:int , seq_len:int , dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #matrix of shape (seq_len,d_model)
        #Positional encoding
        pe = torch.zeroes(seq_len,d_model) 

        #create a vector of shape (seq_len)
        position = torch.range(0 , seq_len , dtype=torch.float).unsqueeze(1) #(seq_len ,1)
        div_term = torch.exp(torch.arange(0,d_model , 2).float() * (-math.log(10000.0)/d_model))

        #apply sine to even position and cosine to odd
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len , d_model)

        self.register_buffer('pe',pe)

    def forward(self ,x):
        x = x+ (self.pe[: , :x.shape[1], :].requires_grad_(False))
        return self.dropout(x)

    
class LayerNormalization(nn.Module):

    def __init__(self , eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeroes(1)) #beta

    def forward(self , x):
        mean = x.mean(dim=-1 ,keepdim = True)
        std = x.std(dim = -1 , keepdim= True)
        return self.alpha * (x-mean) / (self.eps + std) + self.bias


class FeedForward(nn.Module):

    def __init__(self , d_model:int , d_ff:int , dropout:float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model , d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff , d_model) #W2 and B2
        

    def forward(self , x):
        #(Batch, Seq_len , d_model) --> (Batch , Seq_len , d_ff) --> (Batch , Seq_len , d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttenttion(nn.Module):

    def __init__(self , d_model:int , num_heads:int , dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = num_head
        assert d_model % num_heads == 0 , "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model , d_model)
        self.w_k = nn.Linear(d_model , d_model)
        self.w_v = nn.Linear(d_model , d_model)
        self.w_o = nn.Linear(d_model , d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query , key , value , mask , dropout:nn.Dropout):
        d_k = query.shape[-1]


        # (Batch , h , Seq_len , d_k) @ (Batch , h , d_k , Seq_len) --> (Batch , h , Seq_len , Seq_len)
        attention_score = (query @ key.transpose(-2 , -1)) / math.sqrt(d_k) #(B , h , Seq_len , Seq_len)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0 , -1e9) ##replace with a very small value
        attention_score = attention_score.softmax(dim = -1) #(B , h , Seq_len , Seq_len)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value) , attention_score #(B , h , Seq_len , d_k)

    def forward(self , query , key , value , mask):
        batch_size = query.shape[0]

        #linear projections
        Q = self.w_q(query) #(B , Seq_len , d_model)
        K = self.w_k(key)   #(B , Seq_len , d_model)
        V = self.w_v(value) #(B , Seq_len , d_model)

        #split into h heads
        Q = Q.view(Q.shape[0] , Q.shape[1] , self.h , self.d_k).transpose(1,2) #(B , h , Seq_len , d_k)
        K = K.view(K.shape[0] , K.shape[1] , self.h , self.d_k).transpose(1,2) #(B , h , Seq_len , d_k)
        V = V.view(V.shape[0] , V.shape[1] , self.h , self.d_k).transpose(1,2) #(B , h , Seq_len , d_k)

        x, self.attention_score = MultiHeadAttenttion.attention(Q , K , V , mask , self.dropout) #(B , h , Seq_len , d_k)


        # (batch , h , seq_len , d_k) --> (batch , seq_len , h , d_k) --> (batch , seq_len , d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0] , -1 , self.h * self.d_k) #(B , Seq_len , d_model)

        return slef.w_o(x) #(B , Seq_len , d_model)

    
class SublayerConnection(nn.Module):

    def __init__(self , dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self , x , sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self , self_attention_block: MultiHeadAttenttion, feed_forward_block: FeedForward , dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([SublayerConnection(dropout) for _ in range(2)])

    def forward(self , x , src_mask):
        x = self.residual_connections[0](x , lambda x: self.self_attention_block(x , x , x , src_mask))
        x = self.residual_connections[1](x , self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self , layer:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self , x , src_mask):
        for layer in self.layers:
            x = layer(x , src_mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):

    def __init__(self , self_attention_block: MultiHeadAttenttion , cross_attention_block: MultiHeadAttenttion , feed_forward_block: FeedForward , dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([SublayerConnection(dropout) for _ in range(3)])

    def forward(self , x , encoder_output, src_mask , tgt_mask):
        x = self.residual_connections[0](x , lambda x: self.self_attention_block(x , x , x , tgt_mask))
        x = self.residual_connections[1](x , lambda x: self.src_attention_block(x , encoder_output , encoder_output, src_mask))
        x = self.residual_connections[2](x , self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self , layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self , x , encoder_output , src_mask , tgt_mask):
        for layer in self.layers:
            x = layer(x , encoder_output , src_mask , tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self , d_model:int , vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model , vocab_size)

    def forward(self , x):
        #(batch , seq_len , d_model) --> (batch , seq_len , vocab_size)
        return torch.l;og_softmax(self.proj(x) , dim = -1)


class Transformer(nn.Module):

    def __init__(self , encoder:Encoder , decoder:Decoder , src_embed:nn.Module , tgt_embed:nn.Module ,src_pos:PositionalEncoding , tgt_pos:PositionalEncoding ,projection_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection_layer

    def encode(self , src , src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)) , src_mask)
   
    def decode(self , tgt , encoder_output , src_mask , tgt_mask):
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)) , encoder_output , src_mask , tgt_mask)

    def project(self , x):
        return self.projection(x)


def build_transformer(src_vocab_size:int , tgt_vocab_size:int , d_model:int = 512 , d_ff:int = 2048 , num_heads:int = 8 , num_layers:int = 6 , dropout:float = 0.1 , max_seq_len:int = 5000) -> Transformer:
    #Input Embeddings
    src_embed = InputEmbedding(d_model , src_vocab_size)
    tgt_embed = InputEmbedding(d_model , tgt_vocab_size)

    #Positional Encodings
    src_pos = PositionalEncoding(d_model , max_seq_len , dropout)
    tgt_pos = PositionalEncoding(d_model , max_seq_len , dropout)

    #Encoder Layers
    encoder_layers = nn.ModuleList([EncoderLayer(MultiHeadAttenttion(d_model , num_heads , dropout) , FeedForward(d_model , d_ff , dropout) , dropout) for _ in range(num_layers)])
    encoder = Encoder(encoder_layers)

    #Decoder Layers
    decoder_layers = nn.ModuleList([DecoderBlock(MultiHeadAttenttion(d_model , num_heads , dropout) , MultiHeadAttenttion(d_model , num_heads , dropout) , FeedForward(d_model , d_ff , dropout) , dropout) for _ in range(num_layers)])
    decoder = Decoder(decoder_layers)

    #Projection Layer
    projection_layer = ProjectionLayer(d_model , tgt_vocab_size)

    #Transformer Model
    model = Transformer(encoder , decoder , src_embed , tgt_embed , src_pos , tgt_pos , projection_layer)

    return model