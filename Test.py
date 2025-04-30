import cv2
import faiss
import torch
import torch.nn as nn
from MultiRegionModel import EnhancedMultiRegionModel
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from scrfd import SCRFD, Threshold
from PIL import Image

# Khởi tạo face detectorq
face_detector = SCRFD.from_path("Scrfd/scrfd.onnx")
threshold = Threshold(probability=0.4)

# Kiểm tra phiên bản Faiss
print(faiss.__version__)

d = 512  # Chiều của vector embeddings
index = faiss.IndexFlatIP(d)  # Sử dụng Inner Product cho cosine similarity

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Áp dụng transform
transform = Compose([
    ToTensor(),
    Resize((224, 224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Khởi tạo mô hình và tải checkpoint
model = EnhancedMultiRegionModel()
checkpoint = torch.load('save_model/last.pt')
model.load_state_dict(checkpoint['state_dict'])
print(checkpoint['accuracy'])
# Chuyển model vào chế độ đánh giá
model.eval()

# Đọc ảnh đầu tiên (img1)
img1 = cv2.imread('Test_Images/test.png')
imgPIL1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
face1 = face_detector.detect(imgPIL1, threshold=threshold)

# Lấy bounding box của khuôn mặt trong ảnh đầu tiên
bbox1 = face1[0].bbox
x_left1 = int(bbox1.upper_left.x)
y_left1 = int(bbox1.upper_left.y)
x_right1 = int(bbox1.lower_right.x)
y_right1 = int(bbox1.lower_right.y)

# Cắt khuôn mặt từ ảnh đầu tiênq
img1 = img1[y_left1:y_right1, x_left1:x_right1]
cv2.imshow('img2', img1)
cv2.waitKey(0)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Áp dụng transform cho ảnh đầu tiên
img1 = transform(img1)
img1 = img1.unsqueeze(0)  # Thêm batch dimension

# Kiểm tra kích thước của ảnh sau khi transform
print(f"img1 shape: {img1.shape}")

# Chuyển model vào chế độ đánh giá
with torch.no_grad():
    embedding1 = model(img1)

# Chuẩn hóa embedding1
embedding1 = torch.nn.functional.normalize(embedding1)
embedding1 = embedding1.cpu().numpy()

# Thêm embedding vào Faiss index
index.add(embedding1)  # Đảm bảo embedding là NumPy array
distances, indices = index.search(embedding1, k=1)
print(distances, indices)


img2 = cv2.imread('Test_Images/test2.jpeg')
imgPIL1 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
face1 = face_detector.detect(imgPIL1, threshold=threshold)

# Lấy bounding box của khuôn mặt trong ảnh đầu tiên
bbox1 = face1[0].bbox
x_left1 = int(bbox1.upper_left.x)
y_left1 = int(bbox1.upper_left.y)
x_right1 = int(bbox1.lower_right.x)
y_right1 = int(bbox1.lower_right.y)

# Cắt khuôn mặt từ ảnh đầu tiên
img2 = img2[y_left1:y_right1, x_left1:x_right1]
cv2.imshow('img2', img2)
cv2.waitKey(0)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Áp dụng transform cho ảnh đầu tiên
img2 = transform(img2)
img2 = img2.unsqueeze(0)  # Thêm batch dimension

# Kiểm tra kích thước của ảnh sau khi transform
print(f"img1 shape: {img2.shape}")

# Chuyển model vào chế độ đánh giá
with torch.no_grad():
    embedding2 = model(img2)

# Chuẩn hóa embedding1
embedding2 = torch.nn.functional.normalize(embedding2)
embedding2 = embedding2.cpu().numpy()
distances, indices = index.search(embedding2, k=1)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
print('similarity')
print(distances, indices)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi ảnh từ OpenCV (BGR) sang PIL (RGB)
    imgPIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Phát hiện khuôn mặt
    face = face_detector.detect(imgPIL, threshold=threshold)
    if len(face) == 0:
        continue  # Nếu không phát hiện khuôn mặt, tiếp tục vòng lặp

    # Lấy bounding box của khuôn mặt
    bbox = face[0].bbox
    x_left = int(bbox.upper_left.x)
    y_left = int(bbox.upper_left.y)
    x_right = int(bbox.lower_right.x)
    y_right = int(bbox.lower_right.y)
    frame = cv2.rectangle(frame, (x_right, y_right), (x_left, y_left), (255,0,0), 1)

    # Cắt khuôn mặt từ ảnh
    img = frame[y_left:y_right, x_left:x_right]

    # Áp dụng transform cho ảnh webcam
    img = transform(img)
    img = img.unsqueeze(0)  # Thêm batch dimension

    with torch.no_grad():
        embedding2 = model(img)

    # Chuẩn hóa embedding2
    embedding2 = torch.nn.functional.normalize(embedding2)

    # Tìm kiếm cosine similarity trong Faiss
    testEmb = embedding2.cpu().numpy()
    distances, indices = index.search(testEmb, k=1)

    # In ra kết quả
    print(f"Cosine similarity: {distances[0][0]}")
    print(f"Index of the closest match: {indices[0][0]}")

    # Hiển thị kết quả nếu cosine similarity > 0.5
    if distances[0][0] > 0.5:
        frame = cv2.putText(frame, "VVT",(50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    # Hiển thị ảnh kết quả
    cv2.imshow("Face Detected", frame)

    # Dừng chương trình khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
