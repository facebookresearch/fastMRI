import os
import dicom2nifti
import dicom2nifti.settings as settings

# 可选配置：加快转换速度，跳过过度检查
settings.disable_validate_slice_increment()
settings.disable_validate_orthogonal()
settings.disable_validate_slicecount()

def convert_dicom_to_nifti(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        if files:
            try:
                print(f"正在转换：{root}")
                dicom2nifti.convert_directory(
                    root,
                    output_folder,
                    compression=True,
                    reorient=True
                )
            except Exception as e:
                print(f"转换失败 {root}：{e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 DICOM 文件转换为 NIfTI 格式")
    parser.add_argument("input_folder", help="DICOM 文件所在路径")
    parser.add_argument("output_folder", help="输出 NIfTI 文件保存路径")

    args = parser.parse_args()
    convert_dicom_to_nifti(args.input_folder, args.output_folder)
