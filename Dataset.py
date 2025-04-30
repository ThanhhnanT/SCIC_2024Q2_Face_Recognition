import os
from PIL import Image
from scrfd import SCRFD, Threshold
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, root='./Dataset', transform=None):
        self.img_path = os.listdir(root)
        self.images = []
        self.transform = transform
        for path in self.img_path:
            label = int(path.split("_")[0])
            img_path = os.path.join(root, path)
            self.images.append((img_path, label))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

if __name__ == '__main__':
    data = FaceDataset()
    img, label = data[2]
    print('len:', data.__len__())
    labels = [label for _, label in data.images]
    print("Max label:", max(labels))
    print("Min label:", min(labels))

    cv2.imshow("{}".format(label),img)
    cv2.waitKey(0)
