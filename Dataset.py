import os
from PIL import Image
from scrfd import SCRFD, Threshold
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self,root='./lfw',transform = None, face_detector = SCRFD.from_path("./scrfd.onnx"), thres = Threshold(probability=0.4)):
        self.face_det = face_detector
        self.thres = thres
        self.categories = os.listdir(root)
        self.transform = transform
        self.dirs = [os.path.join(root, cate) for cate in self.categories]
        self.images = []
        couunter = 0
        self.labels = []
        for iter, dir in enumerate(self.dirs):
            if not os.path.isdir(dir):
                continue
            images = os.listdir(dir)
            self.images.append((os.path.join(dir, images[0]), iter))
            self.labels.append(iter)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img_cv = cv2.imread(img_path)
        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        face = self.face_det.detect(img, threshold=self.thres)
        bbox = face[0].bbox
        x_left = int( bbox.upper_left.x)
        y_left = int(bbox.upper_left.y)
        x_right = int(bbox.lower_right.x)
        y_right = int(bbox.lower_right.y)
        img_cv = img_cv[y_left:y_right, x_left:x_right]
        if self.transform != None:
            self.transform(img_cv)
        return img_cv, label


if __name__ == '__main__':
    data = FaceDataset()
    img, label = data[32]
    cv2.imshow("dasd", img)
    cv2.waitKey(0)