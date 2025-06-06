# fastMRI 2.0 Dataset

This folder contains manifest info for version 2.0 of the fastMRI dataset,
created on January 30, 2023.

The primary change from version 1.0 is that the data is now **batched** to
facilitate easier download and storage. The batches were designed such that
they all occupy about 100 GB. To inform users of what is inside each batch
prior to download, they can examine *manifest files* in each folder.

Each *manifest file* contains the names of the files in that corresponding
batch. For example, `brain_multicoil_train_batch_0.txt` lists the files
contained in `brain_multicoil_train_batch_0.tar.xz`.

The files have been compressed with the more efficient `xz` command. For
end-users, this doesn't make much of a difference. Your same `tar` commands
should be able to unzip the files. They're just now smaller for download.

Lastly, version 2.0 of the data includes the release of the
**fully-sampled brain test data** (denoted by "brain_multicoil_test_full").
The release of this data was promised as part of the following paper:

Radmanesh, A.\*, Muckley, M. J.\*, Murrell, T., Lindsey, E., Sriram, A., Knoll, F., ... & Lui, Y. W. (2022).
[Exploring the Acceleration Limits of Deep Learning VarNet-based Two-dimensional Brain MRI](https://doi.org/10.1148/ryai.210313). *Radiology: Artificial Intelligence*, 4(6), page e210313.

We made the decision to release the data since, at the time, there was less use
of the brain leaderboard. In addition, the above paper uses the fully-sampled
test set for analysis, so this facilitates reproducing the results in the
paper.

Since the brain test set has been released, the brain leaderboard is now
disabled at fastmri.org.
