import os
import SimpleITK as sitk

def bias_field_correction(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            print(f"正在校正: {input_path}")
            image = sitk.ReadImage(input_path, sitk.sitkFloat32)

            mask = sitk.OtsuThreshold(image, 0, 1, 200)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected = corrector.Execute(image, mask)

            sitk.WriteImage(corrected, output_path)
            print(f"已保存: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对 NIfTI 图像进行 Bias Field 校正")
    parser.add_argument("input_folder", help="输入 NIfTI 文件夹")
    parser.add_argument("output_folder", help="输出校正后文件夹")
    args = parser.parse_args()
    bias_field_correction(args.input_folder, args.output_folder)
