RunInference.py is a standalone script to run inference with the model presented in Johnson. et al, Radiology 2022.

The model can be downloaded from https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/prospective_knee_varnet.pt

To run the model on test data run:

python RunInference.py \
    --data-dir DATA \
    --experiment-dir LOG_DIR \
    --dicom \
    --resume MODEL
