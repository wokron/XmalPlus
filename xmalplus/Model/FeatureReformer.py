import torch
import torch.nn as nn
import numpy as np
from Model.AttentionLayer import FullAttention, AttentionLayer
from Model.Embed import DataEmbedding
import torch.nn.functional as Function
from configs import configs


class FeatureReformer(nn.Module):
    """
    per featureMetric 2 featureVector
    B for batch size
    F for feature size, 158
    L for feature embedding length, 10
    D for feature embedding dim, 768
    in: [B, F, L, D]
    out: [B, F, d]
    """

    def __init__(self):
        super(FeatureReformer, self).__init__()
        self.reform_dim = configs.reform_dim
        self.embedding = DataEmbedding(c_in=configs.embedding_dim, d_model=configs.d_in)
        self.reformLayer = AttentionLayer(attention=FullAttention(), d_model=configs.d_in, n_heads=configs.n_heads,
                                          d_values=configs.d_values)
        self.conv1x1 = nn.Conv1d(in_channels=configs.embedding_length, out_channels=1, kernel_size=1,
                                 stride=1)

    def forward(self, featureMetric):
        """

        :param featureMetric: [B, F, L, D]
        1. 2 [B * F, L, D]
        2， attentionLayer 2 [B * F, L, d_value]
        3. 1 * 1 Conv + Relu to combined temporal info
        """
        B, F, L, D = featureMetric.shape
        featureMetric = featureMetric.reshape(B * F, L, D)
        featureMetric = self.embedding(featureMetric, None)
        attentionMetric, attn = self.reformLayer(featureMetric, featureMetric, featureMetric, None)
        combinedVector = self.conv1x1(attentionMetric).squeeze().reshape(B, F, -1)
        output = Function.gelu(combinedVector)
        return output


if __name__ == '__main__':
    model = FeatureReformer()
    B, F, L, D = 2, 158, 10, 768
    test_input = torch.randn(B, F, L, D)
    test_output = model(test_input)
    print("Output shape:", test_output.shape)
