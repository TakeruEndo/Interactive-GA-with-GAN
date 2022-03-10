import cv2
import torch
from torch.utils.data import Dataset


class FaceDataset(Dataset):

    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

        def get_file_path(x):
            return "../images/{}".format(x)

        self.df["path"] = df["path"].apply(get_file_path)
        self.file_names = df['path'].values
        self.labels = df["target"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.file_names[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image'].float()
        label = torch.tensor(self.labels[index]).long()
        return image, label
