import os
import cv2
from facenet_pytorch import MTCNN
import glob
from PIL import Image, ImageOps

# Đường dẫn tới video và thư mục lưu các khung hình
video_path = 'static/video/sample_happy.mp4'
output_dir = 'media/frames'
def extract_frames(video_path=video_path, output_dir=output_dir):
    files = glob.glob(f'{output_dir}/*')
    for f in files:
        os.remove(f)

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Khởi tạo MTCNN
    mtcnn = MTCNN()

    # Mở video
    cap = cv2.VideoCapture(video_path[1:])
    frame_count = 0
    print("DEBUG extract_frames:", cap.isOpened())
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # if frame_count % 2:
        #     frame_count += 1
        #     continue
        # Chuyển đổi khung hình sang định dạng RGB (MTCNN yêu cầu đầu vào RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Sử dụng MTCNN để nhận diện khuôn mặt
        boxes, probs, _ = mtcnn.detect(frame_rgb, landmarks=True)
        if boxes is not None:
            for i, ((x1, y1, x2, y2), prob) in enumerate(zip(boxes, probs)):
                if prob < 0.95:
                    continue
                # x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                margin = 0.15
                width = x2 - x1
                height = y2 - y1
                x1 = max(0, int(x1 - margin*width))
                y1 = max(0, int(y1 + margin*height))
                x2 = min(frame.shape[1], int(x2))
                y2 = min(frame.shape[0], int(y2))

                # Cắt khuôn mặt từ khung hình bao gồm background
                face_img = frame[y1:y2, x1:x2]
                # Tăng chiều dọc của khuôn mặt để tạo hiệu ứng kéo dãn
                target_width = 224
                target_height = 224  # Tăng chiều dọc lên để tạo hiệu ứng kéo dãn
                face_img_resized = cv2.resize(face_img, (target_width, target_height))
                # Lưu hình ảnh khuôn mặt
                face_filename = os.path.join(output_dir, f'face_{frame_count}.jpg')
                cv2.imwrite(face_filename, face_img_resized)
        frame_count += 1

    cap.release()
    return 1