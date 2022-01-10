"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import h5py
import numpy as np
import pandas as pd


def create_temp_data(path):
    rg = np.random.default_rng(seed=1234)
    max_num_slices = 15
    max_num_coils = 15
    data_splits = {
        "knee_data": [
            "multicoil_train",
            "multicoil_val",
            "multicoil_test",
            "multicoil_challenge",
            "singlecoil_train",
            "singlecoil_val",
            "singlecoil_test",
            "singlecoil_challenge",
        ],
        "brain_data": [
            "multicoil_train",
            "multicoil_val",
            "multicoil_test",
            "multicoil_challenge",
        ],
    }

    enc_sizes = {
        "train": [(1, 128, 64), (1, 128, 49), (1, 150, 67)],
        "val": [(1, 128, 64), (1, 170, 57)],
        "test": [(1, 128, 64), (1, 96, 96)],
        "challenge": [(1, 128, 64), (1, 96, 48)],
    }
    recon_sizes = {
        "train": [(1, 64, 64), (1, 49, 49), (1, 67, 67)],
        "val": [(1, 64, 64), (1, 57, 47)],
        "test": [(1, 64, 64), (1, 96, 96)],
        "challenge": [(1, 64, 64), (1, 48, 48)],
    }

    metadata = {}
    for dataset in data_splits:
        for split in data_splits[dataset]:
            fcount = 0
            (path / dataset / split).mkdir(parents=True)
            encs = enc_sizes[split.split("_")[-1]]
            recs = recon_sizes[split.split("_")[-1]]
            for i in range(len(encs)):
                fname = path / dataset / split / f"file{fcount}.h5"
                num_slices = rg.integers(2, max_num_slices)
                if "multicoil" in split:
                    num_coils = rg.integers(2, max_num_coils)
                    enc_size = (num_slices, num_coils, encs[i][-2], encs[i][-1])
                    recon_size = (num_slices, recs[i][-2], recs[i][-1])
                else:
                    enc_size = (num_slices, encs[i][-2], encs[i][-1])
                    recon_size = (num_slices, recs[i][-2], recs[i][-1])

                data = rg.normal(size=enc_size) + 1j * rg.normal(size=enc_size)

                if split.split("_")[-1] in ("train", "val"):
                    recon = np.absolute(rg.normal(size=recon_size)).astype(
                        np.dtype("<f4")
                    )
                else:
                    mask = rg.integers(0, 2, size=recon_size[-1]).astype(bool)

                with h5py.File(fname, "w") as hf:
                    hf.create_dataset("kspace", data=data.astype(np.complex64))
                    if split.split("_")[-1] in ("train", "val"):
                        hf.attrs["max"] = recon.max()
                        if "singlecoil" in split:
                            hf.create_dataset("reconstruction_esc", data=recon)
                        else:
                            hf.create_dataset("reconstruction_rss", data=recon)
                    else:
                        hf.create_dataset("mask", data=mask)

                enc_size = encs[i]

                enc_limits_center = enc_size[1] // 2 + 1
                enc_limits_max = enc_size[1] - 2

                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max

                metadata[str(fname)] = (
                    {
                        "padding_left": padding_left,
                        "padding_right": padding_right,
                        "encoding_size": enc_size,
                        "recon_size": recon_size,
                    },
                    num_slices,
                )

                fcount += 1

    return path / "knee_data", path / "brain_data", metadata


def create_temp_annotation(path):
    rg = np.random.default_rng(seed=1234)

    annotations_knee = []
    annotations_brain = []
    label_knee = [
        "Meniscus Tear","Displaced Meniscal Tissue","Bone-Subchondral Edema","Bone Lesion",
        "Bone-Fracture/Contusion/Dislocation","ACL High Grade Sprain","ACL Low-Mod Grade Sprain",
        "MCL High Grade Sprain","MCL Low-Mod Grade Sprain","PCL High Grade Sprain",
        "PCL Low-Mod Grade Sprain","LCL Complex High Grade Sprain","LCL Complex Low-Mod Grade Sprain",
        "Cartilage Full Thickness Loss/Defect","Cartilage Partial Thickness Loss/Defect",
        "Joint Effusion","Joint Bodies","Periarticular Cysts","Muscle Strain","Soft Tissue Lesion",
        "Patellar Retinaculum High Grade Sprain","Artifact"]

    label_brain = [
        "Absent Septum Pellucidum","Craniectomy","Craniotomy","Craniotomy with Cranioplasty",
        "Dural Thickening","Edema","Encephalomalacia","Enlarged Ventricles","Extra-Axial Mass",
        "Intraventricular Substance","Likely Cysts","Lacunar Infarct","Mass","Nonspecific Lesion",
        "Nonspecific White Matter Lesion","Normal Variant","Paranasal Sinus Opacification",
        "Pineal Cyst","Possible Artifact","Posttreatment Change","Resection Cavity","Global Ischemia",
        "Small Vessel Chronic White Matter Ischemic Change","Motion Artifact","Possible Demyelinating Disease",
        "Colpocephaly","White Matter Disease","Innumerable Bilateral Focal Brain Lesions",
        "Extra-Axial Collection","Normal for Age"]

    for i in range(17000):
        annotations_knee.append({
            "file": "file" + str(1000000+rg.integers(1,2546)),
            "slice": str(rg.integers(0,45)),
            "study_level": rg.choice(["Yes","No"],1)[0],
            "x": str(rg.integers(0,255)),
            "y" : str(rg.integers(0,255)),
            "width": str(rg.integers(0,255)),
            "height": str(rg.integers(0,255)),
            "label": rg.choice(label_knee,1)[0]
        })

    annotations_knee_df = pd.DataFrame(annotations_knee,columns=annotations_knee[0].keys())
    annotation_knee_csv = f"{path}/knee_annotation.csv"
    annotations_knee_df.to_csv(annotation_knee_csv)
    
    for i in range(17000):
        annotations_brain.append({
            "file": "file" + str(1000000+rg.integers(1,2546)),
            "slice": str(rg.integers(0,45)),
            "study_level": rg.choice(["Yes","No"],1)[0],
            "x": str(rg.integers(0,255)),
            "y" : str(rg.integers(0,255)),
            "width": str(rg.integers(0,255)),
            "height": str(rg.integers(0,255)),
            "label": rg.choice(label_brain,1)[0]
        })
    
    annotations_brain_df = pd.DataFrame(annotations_brain,columns=annotations_brain[0].keys())
    annotation_brain_csv = f"{path}/brain_annotation.csv"
    annotations_brain_df.to_csv(annotation_brain_csv)

    return annotation_knee_csv, annotation_brain_csv