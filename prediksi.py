import cv2
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn.functional as F

# Load model dan label
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # jumlah kelas bunga
model.load_state_dict(torch.load("D:/ObjectDetections/flower/flower_classifier.pth", map_location="cpu"))
model.eval()

class_names = ["Hibiscus", "Rose", "Sunflower"]

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil ukuran frame
    h, w, _ = frame.shape

    # Hitung koordinat crop tengah (untuk prediksi)
    x1, y1 = w // 4, h // 4
    x2, y2 = w * 3 // 4, h * 3 // 4

    # Crop gambar dari kotak tersebut
    crop = frame[y1:y2, x1:x2]
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0)

    # Prediksi
    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        confidence, pred = torch.max(prob, 1)
        label = class_names[pred.item()]
        conf_score = confidence.item() * 100

    # Tampilkan bounding box & label
    text = f"{label}: {conf_score:.2f}%"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # kotak biru
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Tampilkan hasil frame
    cv2.imshow("Flower Classifier", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
