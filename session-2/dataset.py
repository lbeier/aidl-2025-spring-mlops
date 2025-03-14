import os.path

import torch

import pandas as pd
from PIL import Image

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, images_path, labels_path, transform):
        super().__init__()
        self.images_path = images_path
        self.labels = pd.read_csv(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # suite_id: There are totally 100 suites, each created by a volunteer. [1 ... 100]
        # sample_id: Each volunteer created 10 samples. [1 ... 10]
        # code: Each sample contains characters from 0 to 100M (totally 15 Chinese number characters).
        #       This is a code used to identify each character. [1 ... 15]
        # value: Numerical value of each character.
        # character: The actual Chinese character corresponding to one number.

        # Ex: idx=6002 => suite_id=1, sample_id=1, code=1, value=0, character=零
        suite_id, sample_id, code, value, character = self.labels.loc[idx, :]

        # open image
        # <images_path>/input_<suite_id>_<sample_id>_<code>.jpg
        image_path = os.path.join(self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg")
        sample = Image.open(image_path)

        # Apply transformations if provided
        # The returned image should be converted to a tensor.
        if self.transform:
            sample = self.transform(sample)

        # We need to subtract 1 from code because we will use CrossEntropyLoss which requires
        # indices to start at 0 (not 1). So [1 ... 15] => [1 ... 14]
        return sample, code - 1