import os
import dicom2nifti
import dicom2nifti.settings as settings

def convert_dicom_to_nifti(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for root, dirs, files in os.walk(input_folder):
        if files:
            try:
                print(f"Converting: {root}")
                dicom2nifti.convert_directory(root, output_folder, compression=True, reorient=True)
            except Exception as e:
                print(f"Failed to convert {root}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert DICOM to NIfTI")
    parser.add_argument("input_folder", help="Path to folder containing DICOM files")
    parser.add_argument("output_folder", help="Where to save NIfTI files")

    args = parser.parse_args()
    convert_dicom_to_nifti(args.input_folder, args.output_folder)
