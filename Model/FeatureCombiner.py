import torch
import torch.nn as nn
import numpy as np
from Model.AttentionLayer import FullAttention, AttentionLayer
from Model.Embed import DataEmbedding
import torch.nn.functional as Function
from configs import configs


class FeatureCombiner(nn.Module):
    """
    use attention to combine featureVector
    in : [B, F, D]
    """
    def __init__(self):
        super(FeatureCombiner, self).__init__()
        self.combineLayer = AttentionLayer(attention=FullAttention(), d_model=configs.reform_dim, n_heads=configs.combine_n_heads, d_values=configs.combine_d_values)
        self.marker = nn.Linear(configs.mark_feature, 1)

    def forward(self, featureVector):
        """

        :param featureVector: [B, F, D]
        :return: featureMarkVector [B, F]
        1. [B, F, D] attentionLayer 2 [B, F, D]
        """
        B, F, D = featureVector.shape
        featureCombined, attn = self.combineLayer(featureVector, featureVector, featureVector, None)
        featureCombined = featureCombined.reshape(B * F, -1)
        featureMarkVector = self.marker(featureCombined).reshape(B, F, -1).squeeze()
        output = Function.gelu(featureMarkVector)
        return output


if __name__ == '__main__':
    model = FeatureCombiner()
    B, F, D = 2, 158, 256
    test_input = torch.randn(B, F, D)
    test_output = model(test_input)
    print("Output shape:", test_output.shape)


