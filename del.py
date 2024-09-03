from PIL import Image
import os
import numpy as np


# 遍历文件夹
def process_folder(ct_folder, mask_folder):
    for ct_root, _, ct_files in os.walk(ct_folder):
        for ct_file in ct_files:
            if ct_file.lower().endswith('.png'):
                ct_image_path = os.path.join(ct_root, ct_file)
                mask_image_path = os.path.join(mask_folder, ct_file)
                process_images(ct_image_path, mask_image_path)


# 处理CT和mask图片
def process_images(ct_image_path, mask_image_path):
    # ct_image = Image.open(ct_image_path)
    mask_image = Image.open(mask_image_path)

    # ct_pixel_values = np.array(ct_image)
    mask_pixel_values = np.array(mask_image)

    # 检查mask的像素点是否都相同
    unique_mask_values = np.unique(mask_pixel_values)
    if unique_mask_values.size == 1:
        os.remove(ct_image_path)
        os.remove(mask_image_path)
        print(f"Removed {ct_image_path} and its corresponding mask{mask_image_path}.")


# 主函数
def main():
    infection_mask = "D:\pythonProject\MYnet\Dataset\Infection_Mask"
    ct = "D:\pythonProject\MYnet\Dataset\COVID-19-CT-Seg_20cases"

    for i in range(1, 21):
        ct_folder_name = f"{i:02}"
        mask_folder_name = f"{i:02}"

        ct_folder_path = os.path.join(ct, ct_folder_name)
        mask_folder_path = os.path.join(infection_mask, mask_folder_name)

        process_folder(ct_folder_path, mask_folder_path)


if __name__ == "__main__":
    main()
