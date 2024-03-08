import torch
import torch.nn as nn
from xmalplus.Model.FeatureCombiner import FeatureCombiner
from xmalplus.Model.FeatureReformer import FeatureReformer
from xmalplus.configs import configs


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.featureReformer = FeatureReformer()
        self.featureCombiner = FeatureCombiner()
        self.classifier = nn.Linear(configs.feature_size, 1)

    def forward(self, featureMetric):
        featureVector = self.featureReformer(featureMetric)
        MarkVector = self.featureCombiner(featureVector)
        logits = self.classifier(MarkVector).squeeze()
        output = torch.sigmoid(logits)
        return output


if __name__ == '__main__':
    model = Model()
    B, F, L, D = 2, 158, 10, 768
    test_input = torch.randn(B, F, L, D)
    test_output = model(test_input)
    print("Output shape:", test_output.shape)
    print(test_output)