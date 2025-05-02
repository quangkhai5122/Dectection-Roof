import json
import numpy as np
import cv2 
import base64
import io
from PIL import Image
import datascience

for i in range (49, 101):
    json_path = f'Data/json_final/{i}.json'
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

    # Lưu ảnh mask dưới dạng PNG
    mask_output_path = f'Data/labels_final/{i}.png'
    cv2.imwrite(mask_output_path, mask)

    print(f'Ảnh mask đã được lưu tại: {mask_output_path}')