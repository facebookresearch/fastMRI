import numpy as np
import nibabel as nib
import os

def generate_dummy_nifti(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建一个 3D 数据立方体（带简单结构）
    data = np.zeros((160, 160, 160))
    data[40:120, 40:120, 40:120] = 100  # 一个亮的立方体区域

    affine = np.eye(4)  # 标准 affine
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)
    print(f"✅ 模拟 NIfTI 已生成: {output_path}")

if __name__ == "__main__":
    generate_dummy_nifti("nifti_output/fake_mri.nii.gz")
