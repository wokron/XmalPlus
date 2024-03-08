from FeatureExtractor.FeatureExtraction import FeatureExtraction
from FeatureExtractor.GetFeatureMatrix import GetFeatureMatrix


class FeatureExtractor:
    def __init__(self):
        self.featureExtraction = FeatureExtraction()
        self.getFeatureMatrix = GetFeatureMatrix()

    def main(self, Datapath, Apk):
        self.featureExtraction.getFeature(Datapath, Apk, Datapath)
        return self.getFeatureMatrix.getFeaturefromAPK(Datapath, Apk)


if __name__ == '__main__':
    featureExtractor = FeatureExtractor()
    print(featureExtractor.main(Datapath=r"../Data", Apk="2号人事部.apk"))