import os
import cv2
import faiss
import torch
import torch.nn as nn
from MultiRegionModel import EnhancedMultiRegionModel
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from scrfd import SCRFD, Threshold
from PIL import Image
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo face detector
face_detector = SCRFD.from_path("Scrfd/scrfd.onnx")
threshold = Threshold(probability=0.4)

def Vector_Database(root: str) -> None:
    model = EnhancedMultiRegionModel().to(device)
    checkpoint = torch.load('save_model/last.pt', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model accuracy: {checkpoint['accuracy']}")
    model.eval()

    d = 512
    index = faiss.IndexFlatIP(d)

    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images_path = [os.path.join(root, cate) for cate in os.listdir(root)]
    face_labels = []

    for path in images_path:
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgPIL = Image.fromarray(img_rgb)

        # Tách label từ tên ảnh (VD: ./Test_Images/20521451.jpg → 20521451)
        label = os.path.splitext(os.path.basename(path))[0]
        face_labels.append(label)

        faces = face_detector.detect(imgPIL, threshold=threshold)
        if len(faces) == 0:
            continue

        bbox = faces[0].bbox
        x1, y1 = int(bbox.upper_left.x), int(bbox.upper_left.y)
        x2, y2 = int(bbox.lower_right.x), int(bbox.lower_right.y)
        face_img = img_rgb[y1:y2, x1:x2]

        face_tensor = transform(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(face_tensor)
        embedding = torch.nn.functional.normalize(embedding).cpu().numpy().astype('float32')
        index.add(embedding)

    # Sau vòng for
    faiss.write_index(index, 'face_index.faiss')
    with open("face_labels.pkl", "wb") as f:
        pickle.dump(face_labels, f)
    print("Đã lưu FAISS index và label thành công.")


if __name__ == '__main__':
    Vector_Database(root = "./Test_Images")