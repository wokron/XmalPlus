import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from xmalplus.configs import configs
from xmalplus.Model.Model import Model
from xmalplus.utils.path import PACKAGE_ROOT


class FeatureDetector:
    def __init__(self):
        self.model = Model().to(configs.device)
        self.featurePath = PACKAGE_ROOT / "FeatureEmbedding/feature/feature2semantics.txt"
        self.featureList = []
        with open(self.featurePath, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(":")
                if parts and len(parts) > 0:
                    self.featureList.append(parts[0])
        self.cureFeatures = []
        self.cureAssistance = []
        self.standardDistribution = np.load(PACKAGE_ROOT / "distribution/standardDistribution.npy")
        self.model_path = PACKAGE_ROOT / "modelParam/model_epoch_10.pth"

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def pred_and_detect(self, featureMetric):
        self.load_model()
        self.cureAssistance = []
        self.cureFeatures = []
        with torch.no_grad():
            featureMetric = torch.from_numpy(featureMetric).to(configs.device).unsqueeze(0)
            pred_result = self.model(featureMetric).item()
            prompt = ""
            print(f"Result:{pred_result}")
            if pred_result > 0.5:
                print("Begin Interprete Malware")
                featureVector = self.model.featureReformer(featureMetric)
                MarkVector = self.model.featureCombiner(featureVector)
                weights = self.model.classifier.weight.data  # 获取线性层的权重
                # batch_datam [Batch_size, feature_size]
                # 计算贡献度，不包括偏置
                contributions = MarkVector * weights  # 使用广播机制来匹配维度
                # np.save(r"./distribution/exampleNegDistribution.npy", contributions.cpu().numpy()[0])
                shift = (contributions.cpu().numpy() - self.standardDistribution)
                top_quries = list(np.argsort(shift[0])[-1 * configs.cure_query_num:])
                self.cureFeatures = top_quries
                _, attention_weights = self.model.featureCombiner.combineLayer(featureVector, featureVector, featureVector,
                                                                               None)

                attention_weights = attention_weights.squeeze().permute(1, 0, 2).cpu().numpy()  # head, feature_size, feature_size
                # 2 f, h, f
                core_attention_weights = [attention_weights[featureIndex] for featureIndex in self.cureFeatures]  # [h, f] * size
                for core_attention_weight in core_attention_weights:  # h, f
                    assistance = []
                    for head_weight in core_attention_weight:  # f
                        top_assistance = list(np.argsort(head_weight)[-1 * configs.cureFeatureNum // configs.combine_n_heads:])
                        assistance.extend(top_assistance)
                    assistance = list(set(assistance))
                    self.cureAssistance.append(assistance)
                prompt = self.generatePrompt()
        return pred_result, prompt

    def generatePrompt(self):
        example = configs.Example
        name = "test"
        feature1 = self.featureList[self.cureFeatures[0]]
        feature2 = self.featureList[self.cureFeatures[1]]
        feature3 = self.featureList[self.cureFeatures[2]]
        feature4 = self.featureList[self.cureFeatures[3]]
        assistance1 = ", ".join([self.featureList[index] for index in self.cureAssistance[0]])
        assistance2 = ", ".join([self.featureList[index] for index in self.cureAssistance[1]])
        assistance3 = ", ".join([self.featureList[index] for index in self.cureAssistance[2]])
        assistance4 = ", ".join([self.featureList[index] for index in self.cureAssistance[3]])
        example = example.replace("Name", name)
        example = example.replace("Feature1", feature1)
        example = example.replace("Feature2", feature2)
        example = example.replace("Feature3", feature3)
        example = example.replace("Feature4", feature4)
        example = example.replace("Assistance1", assistance1)
        example = example.replace("Assistance2", assistance2)
        example = example.replace("Assistance3", assistance3)
        example = example.replace("Assistance4", assistance4)
        return example










