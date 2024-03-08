import os
import numpy as np
import pandas as pd
import torch.utils.data
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset


class FeatureRead:
    def __init__(self):
        self.data_path = r".\feature\feature2semantics.txt"
        self.csv_path = r".\feature\feature2semantics.csv"
        self.df = pd.DataFrame()
        self.semantic2name = {}

    def readFromRaw(self):
        name_column = []
        semantic_column = []
        with open(self.data_path, 'r', encoding='utf-8') as raw_file:
            maxLength = 0
            for line in raw_file:
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    name_column.append(parts[0].strip())
                    semantic_column.append(parts[1].strip())
                    self.semantic2name[parts[1].strip()] = parts[0].strip()
                    if len(parts[1].split(" ")) > maxLength:
                        maxLength = len(parts[1].split(" "))
        self.df['name'] = name_column
        self.df['semantic'] = semantic_column
        # self.df.to_csv(self.csv_path)
        print(f"max length = {maxLength}")


class semantic_dataset(Dataset):
    def __init__(self):
        self.reader = FeatureRead()
        self.reader.readFromRaw()
        self.modelPath = r".\bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.modelPath)

    def __getitem__(self, index):
        text = self.reader.df.iloc[index]['semantic']
        tokens = self.tokenizer(text, padding='max_length', max_length=10, truncation=True, return_tensors="pt")
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        return text, input_ids.squeeze(), attention_mask.squeeze()

    def __len__(self):
        return len(self.reader.df)


if __name__ == "__main__":
    dataset = semantic_dataset()
    print("Dataset length:", len(dataset))
    for i in range(5):
        print(dataset[i])
