import pandas as pd
import torch.utils.data
from featureProcess import semantic_dataset
import numpy as np
from torch.utils.data import Dataset
from transformers import BertModel


class FeatureEmbedding:
    def __init__(self):
        self.modelPath = r""
        self.dataset = semantic_dataset()
        self.dataLoader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=10,
            shuffle=True,
        )
        self.modelPath = r".\bert-base-uncased"
        self.model = BertModel.from_pretrained(self.modelPath)

    def embedding(self):
        for i, (semantic_texts, input_ids, attention_mask) in enumerate(self.dataLoader):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
                for j, embedding in enumerate(embeddings):
                    name_replace = self.dataset.reader.semantic2name[semantic_texts[j]].replace("/", "-").replace(">", "-")
                    file_path = f".\embedding\embedding_{name_replace}.pt"
                    torch.save(embedding, file_path)


if __name__ == '__main__':
    featureEmbedding = FeatureEmbedding()
    featureEmbedding.embedding()
