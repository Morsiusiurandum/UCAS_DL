import torch
import torch.nn as nn

from layers import TransformerDecoderLayer, TransformerEncoderLayer


class BERTEncoder(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 model_dimension,
                 maximum_sequence_length,
                 number_of_layers,
                 number_of_attention_heads,
                 hidden_dimension,
                 dropout_rate=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocabulary_size, model_dimension)
        self.position_embedding = nn.Embedding(maximum_sequence_length, model_dimension)
        self.segment_embedding = nn.Embedding(2, model_dimension)

        self.dropout = nn.Dropout(dropout_rate)

        self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(model_dimension, number_of_attention_heads, hidden_dimension, dropout_rate)
                for _ in range(number_of_layers)
        ])

    def forward(self, input_token_ids, segment_token_type_ids, attention_mask=None):
        # position_ids = torch.arange(0, input_token_ids.size(1)).unsqueeze(0).to(input_token_ids.device)
        position_ids = torch.arange(0, input_token_ids.size(1), device=input_token_ids.device, dtype=torch.long).unsqueeze(0).expand(input_token_ids.size(0), -1)
        token_emb = self.token_embedding(input_token_ids)

        pos_emb = self.position_embedding(position_ids)

        seg_emb = self.segment_embedding(segment_token_type_ids)

        embeddings = token_emb + pos_emb + seg_emb
        output = self.dropout(embeddings)

        for layer in self.encoder_layers:
            output = layer(output, attention_mask)
        return output


class BERTDecoder(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 model_dimension,
                 maximum_sequence_length,
                 number_of_layers,
                 number_of_attention_heads,
                 hidden_dimension,
                 dropout_rate=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, model_dimension)
        self.position_embedding = nn.Embedding(maximum_sequence_length, model_dimension)

        self.decoder_layers = nn.ModuleList([
                TransformerDecoderLayer(model_dimension, number_of_attention_heads, hidden_dimension, dropout_rate)
                for _ in range(number_of_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(model_dimension, vocabulary_size)

    def forward(self, target_token_ids, encoder_output, target_mask=None, encoder_mask=None):
        position_ids = torch.arange(0, target_token_ids.size(1)).unsqueeze(0).to(target_token_ids.device)
        token_emb = self.token_embedding(target_token_ids)
        pos_emb = self.position_embedding(position_ids)
        embeddings = token_emb + pos_emb
        output = self.dropout(embeddings)

        for layer in self.decoder_layers:
            output = layer(output, encoder_output, target_mask, encoder_mask)
        return self.output_projection(output)


class BERT2BERTTranslationModel(nn.Module):
    def __init__(self,
                 source_vocabulary_size,
                 target_vocabulary_size,
                 model_dimension=512,
                 maximum_sequence_length=512,
                 number_of_layers=6,
                 number_of_attention_heads=8,
                 hidden_dimension=2048,
                 dropout_rate=0.1):
        super().__init__()
        self.encoder = BERTEncoder(source_vocabulary_size, model_dimension, maximum_sequence_length, number_of_layers, number_of_attention_heads,
                                   hidden_dimension, dropout_rate)
        self.decoder = BERTDecoder(target_vocabulary_size, model_dimension, maximum_sequence_length, number_of_layers, number_of_attention_heads,
                                   hidden_dimension, dropout_rate)

    def forward(self, input_token_ids, segment_token_type_ids, input_attention_mask, target_token_ids, target_attention_mask=None):
        encoder_output = self.encoder(input_token_ids, segment_token_type_ids, input_attention_mask)
        output_logits = self.decoder(target_token_ids, encoder_output, target_attention_mask, input_attention_mask)
        return output_logits
