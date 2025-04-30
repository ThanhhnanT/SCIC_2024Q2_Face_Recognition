
import os
from ArcFaceLoss import SubCenterArcFaceLoss
from Dataset import FaceDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomRotation, ColorJitter
import cv2
import numpy as np
import random
from tqdm import tqdm
from torch.optim import Adam, SGD
from MultiRegionModel import EnhancedMultiRegionModel
from argparse import ArgumentParser

def get_arg():
    parser = ArgumentParser(description='Train a CNN model')
    parser.add_argument("--epoch", "--e", type = int, default=100, help='Number of epoch')
    parser.add_argument("--batch_size", '-b', type=int, default=8, help="batch size")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="logging level")
    parser.add_argument("--train_model", "-t", type=str, default="train_model", help="train_model")
    parser.add_argument("--checkpoint", "-c", type=str, default="./save_model/last.pt")
    args = parser.parse_args()
    return args

def Train():
    arg = get_arg()
    arcface = SubCenterArcFaceLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_transform = Compose(
        [
            ToTensor(),
            RandomRotation(degrees=5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            Resize((224, 224)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = Compose(
        [
            ToTensor(),
            Resize((224, 224)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_data = FaceDataset(transform = train_transform)
    test_data = FaceDataset(transform = test_transform)
    train_dataloader = DataLoader(
        dataset=train_data,
        num_workers=8,
        batch_size=arg.batch_size,
        drop_last= True
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        num_workers=8,
        batch_size=arg.batch_size,
        drop_last=True,
    )
    model = EnhancedMultiRegionModel().to(device)
    checkpoint = torch.load(arg.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    arcface.load_state_dict(checkpoint['arcface_state_dict'])
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochs = arg.epoch
    max = 0
    for epoch in range(epochs):
        process = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        model.train()  # Đặt model vào chế độ huấn luyện
        for iter, (img, label) in enumerate(process):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            embeddings = model(img)
            logits = arcface(embeddings, label)  # Cập nhật hàm ArcFaceLoss nếu cần
            loss = loss_function(logits, label)  # Tính toán loss
            loss.backward()
            optimizer.step()

            # Cập nhật mô tả tiến trình để hiển thị loss
            process.set_postfix(loss=loss.item())

        # Kiểm tra mô hình trên tập kiểm tra
        model.eval()  # Đặt model vào chế độ kiểm tra
        correct = 0
        total = 0

        with torch.no_grad():  # Không tính gradient trong quá trình kiểm tra
            for img, label in test_dataloader:
                img = img.to(device)
                label = label.to(device)

                embeddings = model(img)
                logits = arcface(embeddings, label)
                # print('logit', logits)
                _, predicted = torch.max(logits, 1)
                # print('label',label, label.size(0))
                # print('predictr', predicted)
                total += label.size(0)
                # print('total', total)
                correct += (predicted == label).sum().item()
                # print('correct', correct)
        accuracy = correct / total * 100
        print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {accuracy:.2f}%")
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'arcface_state_dict': arcface.state_dict(),
            'accuracy' : accuracy
        }
        torch.save(checkpoint, os.path.join('save_model', 'last.pt'))
        if accuracy > max:
            max= accuracy
            torch.save(checkpoint, os.path.join('save_model', 'best.pt'))


if __name__ == '__main__':
    Train()