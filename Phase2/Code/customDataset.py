import os
import pandas as pd
from torchvision.io import read_image
import cv2
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):

        with open(annotations_file, 'r') as f:
            self.img_labels = [int(line.strip()) for line in f.readlines()]
        
        # self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform= transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir,f"{idx + 1}.png")
        # img_path = img_path + ".png"
        image = cv2.imread(img_path)
        label = int(self.img_labels[idx])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

