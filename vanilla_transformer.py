

import torch
import torch.nn as nn
import math

# Embedding layer
class InputEmbeddings(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim) # [B, Seq_len] => [B, Seq_len, Emb_dim]

# Positional Encoding

class PositionEncoding(nn.Module):

    def __init__(self, embedding_dim, sequence_len, dropout=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(sequence_len, embedding_dim)
        # Создадим вектор, который будет кодировать позицию вектора в предложении: shape [seq_len, 1]
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # [1, seq_len, vocab_size] Добавляем ось для батча.

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

# Layer Normalization

class LayerNormalization(nn.Module):

    def __init__(self, features, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features)) # Обучаемый параметр, отвечающий за стандартное отклонение
        self.mu = nn.Parameter(torch.zeros(features)) # # Обучаемый параметр, отвечающий за смещение

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.mu

class FeedForwardBlock(nn.Module):

    def __init__(self, embedding_dim, ff_dim, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, n_heads, dropout=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        assert embedding_dim % n_heads == 0, 'embedding_dim is not divisible by n_heads'
        self.d_k = self.embedding_dim // n_heads

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.W_q(q) # [Batch_size, Seq_len, Emb_dim] => [Batch_size, Seq_len, Emb_dim]
        key = self.W_k(k) # [Batch_size, Seq_len, Emb_dim] => [Batch_size, Seq_len, Emb_dim]
        value = self.W_v(v) # [Batch_size, Seq_len, Emb_dim] => [Batch_size, Seq_len, Emb_dim]

        # [Batch_size, Seq_len, Emb_dim] => [Batch_size, n_heads, Seq_len, d_k]
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        return self.W_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, features, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layernorm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, features, self_attention_block, feed_forward_block, dropout=0.5):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, features, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layernorm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layernorm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout=0.5):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, features, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layernorm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.layernorm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, embedding_dim: int, vocabulary_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, x):
        return self.projection(x)


class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embeddings: InputEmbeddings,
                 tgt_embeddings: InputEmbeddings,
                 src_position: PositionEncoding,
                 tgt_position: PositionEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.tgt_embeddings = tgt_embeddings
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_outputs, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embeddings(tgt)
        tgt = self.tgt_position(tgt)
        return self.decoder(tgt, encoder_outputs, src_mask, tgt_mask)

    def projection(self, x):
        return self.projection_layer(x)



def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      embedding_dim=512,
                      n_layers=6,
                      n_heads=8,
                      dropout=0.1,
                      ff_dim=2048) -> Transformer:
    src_embeddings = InputEmbeddings(src_vocab_size, embedding_dim)
    tgt_embeddings = InputEmbeddings(tgt_vocab_size, embedding_dim)

    src_pos = PositionEncoding(embedding_dim, src_seq_len, dropout)
    tgt_pos = PositionEncoding(embedding_dim, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention_block = MultiHeadAttention(embedding_dim, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim, ff_dim, dropout)
        encoder_block = EncoderBlock(embedding_dim, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(n_layers):
        decoder_self_attention_block = MultiHeadAttention(embedding_dim, n_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttention(embedding_dim, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(embedding_dim, ff_dim, dropout)
        decoder_block = DecoderBlock(embedding_dim, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(embedding_dim, nn.ModuleList(encoder_blocks))
    decoder = Decoder(embedding_dim, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(embedding_dim, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embeddings, tgt_embeddings, src_pos, tgt_pos, projection_layer)

    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer
