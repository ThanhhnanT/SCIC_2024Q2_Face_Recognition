from facenet_pytorch import MTCNN
from PIL import Image
import torch

class FaceRegionDataset(torch.utils.data.Dataset):
    def __init__(self, image_path='people.jpg', transform=None):
        self.image_paths = [image_path]
        self.transform = transform
        self.mtcnn = MTCNN(keep_all=True)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)

        if boxes is None:
            return []

        results = []
        for i in range(len(boxes)):
            box = boxes[i]
            landmark = landmarks[i]

            eye_region = self.crop_eye_region(img, landmark)
            forehead_region = self.crop_forehead(img, landmark)
            full_face = self.crop_face(img, box)

            if self.transform:
                eye_region = self.transform(eye_region)
                forehead_region = self.transform(forehead_region)
                full_face = self.transform(full_face)

            results.append({
                'eye': eye_region,
                'forehead': forehead_region,
                'fullface': full_face
            })

        return results

    def crop_eye_region(self, img, landmark):
        left_eye = landmark[0]
        right_eye = landmark[1]
        x1 = int(min(left_eye[0], right_eye[0]) - 10)
        y1 = int(min(left_eye[1], right_eye[1]) - 10)
        x2 = int(max(left_eye[0], right_eye[0]) + 10)
        y2 = int(max(left_eye[1], right_eye[1]) + 10)
        return img.crop((x1, y1, x2, y2))

    def crop_forehead(self, img, landmark):
        left_eye = landmark[0]
        right_eye = landmark[1]
        center_eye_x = (left_eye[0] + right_eye[0]) / 2
        center_eye_y = (left_eye[1] + right_eye[1]) / 2

        nose = landmark[2]
        h = abs(center_eye_y - nose[1])
        x1 = int(center_eye_x - h)
        y1 = int(center_eye_y - 2 * h)
        x2 = int(center_eye_x + h)
        y2 = int(center_eye_y - h / 2)
        return img.crop((x1, y1, x2, y2))

    def crop_face(self, img, box):
        x1, y1, x2, y2 = [int(b) for b in box]
        return img.crop((x1, y1, x2, y2))

if __name__ == '__main__':
    dataset = FaceRegionDataset()
    results = dataset[0]

    for i, face in enumerate(results):
        print(f"Face {i+1}:")
        face['eye'].show(title=f"Eye Region {i+1}")
        face['forehead'].show(title=f"Forehead Region {i+1}")
        face['fullface'].show(title=f"Full Face {i+1}")
