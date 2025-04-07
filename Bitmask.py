import os
import json
import numpy as np
import cv2 
import base64
import io
from PIL import Image

json_folder = "D:\\TTNT 2025\\Data\\json_vn"
output_mask_folder = "D:\\TTNT 2025\\Data\\labels_vn"
image_folder = "D:\\TTNT 2025\\Data\\images_vn"

os.makedirs(output_mask_folder, exist_ok=True)

json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Nếu file JSON có trường "imageData", decode ảnh từ đó,
    # Không thì sử dụng "imagePath" để load ảnh gốc.
    if data.get('imageData'):
        image_data = data['imageData']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image = np.array(image)
    else:
        image_path = data.get('imagePath')
        image = cv2.imread(image_path)

    # Lấy kích thước ảnh (2 gt cuối)
    height, width = image.shape[:2]

    # Tạo mảng mask đen (0) với kích thước ảnh gốc
    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data['shapes']:
        if shape['label'].lower() == 'roof':
            points = np.array(shape['points'], dtype=np.float32)
            points = np.round(points).astype(np.int32)
            points = points.reshape((-1, 1, 2))
            # Vẽ polygon đầy đủ trên mask với màu trắng (255)
            cv2.fillPoly(mask, [points], color=255)
            
    mask = mask.reshape((height, width, 1))

    output_filename = os.path.splitext(json_file)[0] + ".png"
    output_path = os.path.join(output_mask_folder, output_filename)

    cv2.imwrite(output_path, mask)
    print(f"Đã lưu mask: {output_path}")