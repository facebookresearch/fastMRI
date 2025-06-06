import os
import nibabel as nib
import numpy as np

def extract_roi(input_folder, output_folder, roi_size=(80, 80, 80)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            filepath = os.path.join(input_folder, filename)
            img = nib.load(filepath)
            data = img.get_fdata()

            center = np.array(data.shape) // 2
            half = np.array(roi_size) // 2

            start = center - half
            end = center + half

            roi = data[
                start[0]:end[0],
                start[1]:end[1],
                start[2]:end[2]
            ]

            roi_img = nib.Nifti1Image(roi, img.affine, img.header)
            save_path = os.path.join(output_folder, filename)
            nib.save(roi_img, save_path)

            print(f"✅ 已提取 ROI：{save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从 NIfTI 图像中提取 ROI 区域")
    parser.add_argument("input_folder", help="输入偏场校正的 NIfTI 文件夹")
    parser.add_argument("output_folder", help="保存 ROI 提取结果的文件夹")
    args = parser.parse_args()

    extract_roi(args.input_folder, args.output_folder)
