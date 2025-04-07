import os
import imageio

folder_path = "D:\\TTNT 2025\\Data\\labels_vn\\"  

file_list = os.listdir(folder_path)

for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    try:
        img = imageio.imread(file_path)
        print(f"{filename}: {img.shape}")
    except Exception as e:
        print(f"Không đọc được {filename}: {e}")
