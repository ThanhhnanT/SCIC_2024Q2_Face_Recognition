import cv2
import faiss
import torch
from BuildModel.MultiRegionModel import EnhancedMultiRegionModel
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from scrfd import SCRFD, Threshold
from PIL import Image
import pickle

# Khởi tạo face detectorq
face_detector = SCRFD.from_path("Scrfd/scrfd.onnx")
threshold = Threshold(probability=0.4)

# Kiểm tra phiên bản Faiss
print(faiss.__version__)

index = faiss.read_index("face_index.faiss")  # Sử dụng Inner Product cho cosine similarity
with open("VectorDatabase/face_labels.pkl", "rb") as f:
    name = pickle.load(f)
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
checkpoint = torch.load('Save_model/last.pt')
model.load_state_dict(checkpoint['state_dict'])
print(checkpoint['accuracy'])
# Chuyển model vào chế độ đánh giá
model.eval()


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
    distances, indices = index.search(testEmb, k=4)

    # In ra kết quả
    print(f"Cosine similarity: {distances}")
    print(f"Index of the closest match: {indices}")
    for idx in indices[0]:
        print("Name_img: ", name[idx])
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
