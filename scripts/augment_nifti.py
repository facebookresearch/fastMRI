import os
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate, shift

def augment_nifti(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
            continue

        filepath = os.path.join(input_folder, filename)
        img = nib.load(filepath)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        # 保存原图
        nib.save(nib.Nifti1Image(data, affine, header), os.path.join(output_folder, f"{filename}"))

        # ➕ 左右翻转
        flipped = np.flip(data, axis=0)
        nib.save(nib.Nifti1Image(flipped, affine, header), os.path.join(output_folder, f"{filename[:-7]}_flipx.nii.gz"))

        # ➕ 上下翻转
        flipped_y = np.flip(data, axis=1)
        nib.save(nib.Nifti1Image(flipped_y, affine, header), os.path.join(output_folder, f"{filename[:-7]}_flipy.nii.gz"))

        # ➕ 旋转 90 度（z 轴）
        rotated = rotate(data, 90, axes=(0, 1), reshape=False)
        nib.save(nib.Nifti1Image(rotated, affine, header), os.path.join(output_folder, f"{filename[:-7]}_rot90.nii.gz"))

        # ➕ 平移 (x, y, z)
        shifted = shift(data, shift=(5, -5, 3))
        nib.save(nib.Nifti1Image(shifted, affine, header), os.path.join(output_folder, f"{filename[:-7]}_shift.nii.gz"))

        print(f"✅ 已增强：{filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对 NIfTI 图像执行数据增强")
    parser.add_argument("input_folder", help="输入 NIfTI 文件夹（如 ROI）")
    parser.add_argument("output_folder", help="输出增强文件夹")
    args = parser.parse_args()

    augment_nifti(args.input_folder, args.output_folder)
