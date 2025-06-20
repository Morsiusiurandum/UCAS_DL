import math

import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, model_dimension: int, number_of_heads: int, dropout_rate: float = 0.1):
        """
        多头注意力层，包含查询、键、值的线性变换和输出的线性变换。
        :param model_dimension: 模型的维度，通常是词嵌入的维度。
        :param number_of_heads: 头的个数。
        :param dropout_rate: dropout的比率，dropout在softmax后，乘value之前。
        """
        super().__init__()
        assert model_dimension % number_of_heads == 0
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads

        self.query_projection = nn.Linear(model_dimension, model_dimension)
        self.key_projection = nn.Linear(model_dimension, model_dimension)
        self.value_projection = nn.Linear(model_dimension, model_dimension)

        self.output_projection = nn.Linear(model_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)

        def split_heads(x):
            """
            将输入x分割成多个头部
            """
            return x.view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)

        query = split_heads(self.query_projection(query))
        key = split_heads(self.key_projection(key))
        value = split_heads(self.value_projection(value))

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        if attention_mask is not None:
            # 保证 attention_mask 可广播到 attention_scores
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(self.dropout(attention_weights), value)

        # 将注意力输出重新组合成原始形状
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.number_of_heads * self.head_dimension)
        return self.output_projection(attention_output)


class FeedForwardLayer(nn.Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
                nn.Linear(model_dimension, hidden_dimension),
                nn.ReLU(),
                nn.Linear(hidden_dimension, model_dimension),
                nn.Dropout(dropout_rate)
        )

    def forward(self, input_tensor):
        return self.network(input_tensor)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层，自注意力->前馈网络。
    """

    def __init__(self, model_dimension: int, number_of_heads: int, hidden_dimension: int, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(model_dimension, number_of_heads, dropout_rate)
        self.feed_forward = FeedForwardLayer(model_dimension, hidden_dimension, dropout_rate)
        self.layer_normalization_1 = nn.LayerNorm(model_dimension)
        self.layer_normalization_2 = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tensor, attention_mask=None):
        attention_output = self.self_attention(input_tensor, input_tensor, input_tensor, attention_mask)
        input_tensor = self.layer_normalization_1(input_tensor + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(input_tensor)
        return self.layer_normalization_2(input_tensor + self.dropout(feed_forward_output))


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层，自注意力->交叉注意力->前馈网络。
    """

    def __init__(self, model_dimension: int, number_of_heads: int, hidden_dimension: int, dropout_rate: float = 0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttentionLayer(model_dimension, number_of_heads, dropout_rate)
        self.cross_attention = MultiHeadAttentionLayer(model_dimension, number_of_heads, dropout_rate)
        self.feed_forward = FeedForwardLayer(model_dimension, hidden_dimension, dropout_rate)
        self.layer_normalization_1 = nn.LayerNorm(model_dimension)
        self.layer_normalization_2 = nn.LayerNorm(model_dimension)
        self.layer_normalization_3 = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, target_tensor, encoder_output, target_mask=None, encoder_mask=None):
        self_attention_output = self.masked_self_attention(target_tensor, target_tensor, target_tensor, target_mask)
        target_tensor = self.layer_normalization_1(target_tensor + self.dropout(self_attention_output))

        cross_attention_output = self.cross_attention(target_tensor, encoder_output, encoder_output, encoder_mask)
        target_tensor = self.layer_normalization_2(target_tensor + self.dropout(cross_attention_output))

        feed_forward_output = self.feed_forward(target_tensor)
        return self.layer_normalization_3(target_tensor + self.dropout(feed_forward_output))
