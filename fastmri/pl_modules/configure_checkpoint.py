import os
import pytorch_lightning as pl


def configure_checkpoint(
    default_root_dir, checkpoint_name="checkpoints", monitor="val_loss", mode="min"
):
    """
    Checkpoint configuration function.

    This simple function sets up a pl.callbacks.ModelCheckpoint for logging
    into default_root_dir / checkpoint_name. Prior to setting up the checkpoint
    callback, it checks the directory for any existing checkpoints and returns
    the most recent one. If default_root_dir / checkpoint_name does not exist,
    then this function creates it.

    Args:
        default_root_dir (pathlib.Path): Default root directory for logging.
        checkpoint_name (str, optional): Name for checkpoint subfolder.
            Defaults to "checkpoints".
        monitor (str, optional): Quantity to monitor for saving checkpoints.
            Defaults to "val_loss".
        mode (str, optional): Mode for monitoring. Defaults to "min".
    
    Returns:
        (tuple): Tuple containing:
            pl.callbacks.ModelCheckpoint: A ModelCheckpooint for a PyTorch
                Lightning Trainer.
            pathlib.Path: A checkpoint you can pass to resume_from_checkpoint
                in the trainer to resume existing training. If no checkpoint is
                found, this returns None.
    """
    checkpoint_dir = default_root_dir / checkpoint_name

    resume_from_checkpoint = None
    if checkpoint_dir.exists():
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            resume_from_checkpoint = ckpt_list[-1]
    else:
        checkpoint_dir.mkdir()  # note: better to create this outside main

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=default_root_dir / checkpoint_name,
        save_top_k=True,
        verbose=True,
        monitor=monitor,
        mode=mode,
        prefix="",
    )

    return checkpoint_callback, resume_from_checkpoint
