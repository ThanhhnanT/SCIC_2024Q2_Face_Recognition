import cv2
from scrfd import SCRFD, Threshold
import os
from PIL import Image

def SetUpData():
    root = './img_align_celeba'
    list = [os.path.join(root, i) for i in os.listdir(root)]
    face_detec = SCRFD.from_path("Scrfd/scrfd.onnx")
    thres = Threshold(probability=0.5)
    print(len(list))
    iter = 0
    for  img_path in list:
        ori_img = cv2.imread(img_path)
        if ori_img is None:
            print(f"❌ Không thể đọc ảnh: {img_path}")
            continue  # Bỏ qua ảnh bị lỗi
        print(img_path)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        faces = face_detec.detect(img, threshold=thres)
        if len(faces) ==0 :
            continue
        bbox = faces[0].bbox
        x_left = int(bbox.upper_left.x)
        y_left = int(bbox.upper_left.y)
        x_right = int(bbox.lower_right.x)
        y_right = int(bbox.lower_right.y)
        # Cắt khuôn mặt
        face_img = ori_img[y_left:y_right, x_left:x_right]

        # Tạo bản sao ảnh khuôn mặt
        img_cover_mouth = face_img.copy()
        img_cover_eyes = face_img.copy()
        img_cover_head = face_img.copy()
        # === Che miệng ===
        # Giả định vùng miệng nằm ở 2/3 phía dưới khuôn mặt
        height, width = face_img.shape[:2]
        mouth_top = int(height * 1 / 2)
        mouth_bottom = int(height * 9 / 10)
        cv2.rectangle(img_cover_mouth, (0, mouth_top), (width, mouth_bottom), (0, 0, 0), -1)

        # === Che mắt ===
        # Giả định vùng mắt nằm ở 1/4 đến 1/2 khuôn mặt
        eye_top = int(height * 1 / 5)
        eye_bottom = int(height * 3 / 5)
        cv2.rectangle(img_cover_eyes, (0, eye_top), (width, eye_bottom), (0, 0, 0), -1)

        cnt =1
        # Lưu hoặc hiển thị ảnh
        cv2.imwrite("Dataset/{}_{}.jpeg".format(iter, cnt), face_img)

        cnt +=1
        cv2.imwrite("Dataset/{}_{}.jpeg".format(iter, cnt), img_cover_eyes)
        cnt += 1

        cv2.imwrite("Dataset/{}_{}.jpeg".format(iter, cnt), img_cover_mouth)
        iter +=1


if __name__ == '__main__':
    SetUpData()