from xmalplus.FeatureExtractor.FeatureExtractor import FeatureExtractor
from xmalplus.featureDetect import FeatureDetector
from xmalplus.FeatureEmbedding.embeddingProcesss import EmbeddingProcess
from xmalplus.configs import configs
import numpy as np


class Exp:
    def __init__(self):
        self.featureExtractor = FeatureExtractor()
        self.featureDetector = FeatureDetector()
        self.Adapter = EmbeddingProcess(r"./FeatureEmbedding/embedding")
        self.Adapter.process()

    def main(self, Datapath, Apk):
        featureVector = self.featureExtractor.main(Datapath=Datapath, Apk=Apk)
        featureMetrix = np.zeros((len(featureVector), 10, 768), dtype=np.float32)
        for i, feature in enumerate(featureVector):
            if feature > 0:
                featureMetrix[i] = self.Adapter.name2numpy[
                    configs.cols_featureList[i].replace("/", "-").replace(">", "-")]
        result, prompt = self.featureDetector.pred_and_detect(featureMetrix)
        print(result)
        print(prompt)


if __name__ == '__main__':
    exp = Exp()
    # exp.main(r"./Data", "2号人事部.apk")  # Pos Example
    # exp.main(r"./Data", "06eb19d137ff0a0ccd778b236e8d45c3b9b078115b2ed69baf06aee0244980c1.apk")  # Neg Example
