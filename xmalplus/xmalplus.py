from xmalplus.FeatureExtractor.FeatureExtractor import FeatureExtractor
from xmalplus.featureDetect import FeatureDetector
from xmalplus.FeatureEmbedding.embeddingProcesss import EmbeddingProcess
from xmalplus.configs import configs
import numpy as np

from xmalplus.utils.path import PACKAGE_ROOT


class XmalPlus:
    def __init__(self):
        self.featureExtractor = FeatureExtractor()
        self.featureDetector = FeatureDetector()
        self.Adapter = EmbeddingProcess(PACKAGE_ROOT / "FeatureEmbedding/embedding")
        self.Adapter.process()

    def run(self, Datapath, Apk):
        featureVector = self.featureExtractor.run(Datapath=Datapath, Apk=Apk)
        featureMetrix = np.zeros((len(featureVector), 10, 768), dtype=np.float32)
        for i, feature in enumerate(featureVector):
            if feature > 0:
                featureMetrix[i] = self.Adapter.name2numpy[
                    configs.cols_featureList[i].replace("/", "-").replace(">", "-")
                ]
        result, key_features = self.featureDetector.pred_and_detect(featureMetrix)
        return {
            "apk_name": Apk,
            "malware_score": result,
            "key_features": key_features,
        }
