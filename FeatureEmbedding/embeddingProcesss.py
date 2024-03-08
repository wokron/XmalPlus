import os
import torch
import numpy as np


class EmbeddingProcess:
    def __init__(self, root_path):
        self.root_path = root_path
        self.name2numpy = {}

    def process(self):
        files = os.listdir(self.root_path)
        for i, file in enumerate(files):
            file_path = os.path.join(self.root_path, file)
            name = file.replace("embedding_", "").replace(".pt", "")
            # print(name)
            try:
                tensor = torch.load(file_path)
                self.name2numpy[name] = tensor.numpy()
            except EOFError:
                print(name)


if __name__ == '__main__':
    root_path = r"./embedding"
    embeddingProcess = EmbeddingProcess(root_path)
    embeddingProcess.process()
