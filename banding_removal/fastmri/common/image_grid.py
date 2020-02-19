import math
import numpy as np
import logging
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from torch.nn import functional as F
import pathlib
import sys
import textwrap
import pdb
import torch

def grid(image_blocks, losses=None, runinfo=None, header_height=300):
    """
        Pass in a list of image blocks i.e. sets of images to be viewed side-by-side
        in the grid. A header with runinfo is added.
    """
    nimages = len(image_blocks)
    assert nimages != 0

    images_per_block = len(image_blocks[0])
    assert images_per_block != 0

    # Assume all images have the same width and height.
    height = image_blocks[0][0].shape[1]
    width = image_blocks[0][0].shape[2]

    caption_height = 20
    grid_width = 1170

    pad_width = width + 2
    pad_height = height + 2
    block_width = images_per_block * pad_width
    if block_width > grid_width:
        grid_width = block_width

    grid_block_width = grid_width // block_width

    grid_block_height = int(math.ceil(nimages / grid_block_width))
    grid_height = grid_block_height * (pad_height + caption_height)

    def block_location(i):
        row = header_height + (pad_height + caption_height) * (i // grid_block_width)
        col = block_width * int(i % grid_block_width)
        return row, col

    ar = np.zeros([header_height + grid_height,  grid_width, 3])
    for i in range(nimages):
        row, col = block_location(i)

        # Blue border around first image (this could correspond to ground truth)
        ar[row:(row+height+2), col:(col+width+2), 2] = 0.8
        for image in image_blocks[i]:
            ar[(row+1):(row+height+1), (col+1):(col+width+1), :] = image[0, :, :][..., None]
            col += pad_width

    ar *= 255
    ar = np.clip(ar,0,255)

    img_pil = Image.fromarray(ar.astype('uint8'), mode='RGB')

    if runinfo is None:
        return img_pil

    ### Header part
    header_txt = str(runinfo["args"])
    text_width = 160
    header_txt = textwrap.fill(header_txt, width=text_width)
    if len(header_txt) > text_width*9:
        header_txt = header_txt[:text_width*10]

    try:
        header_txt += f"\n Current epoch {runinfo['at_epoch']}"
        #pdb.set_trace()
        dev_losses = runinfo["dev_losses"]
        if len(dev_losses) < 5:
            indexes = range(len(dev_losses))
        elif len(dev_losses) < 25:
            indexes = range(0, len(dev_losses), 5)
        else:
            indexes = range(0, len(dev_losses), 10)
        indexes = list(indexes)

        current = len(dev_losses)-1
        if indexes[-1] != current:
            indexes.append(current)

        for i in indexes:
            dl = dev_losses[i]
            caption = ""
            for k, v in dl.items():
                loss = v
                caption += f" {k}: {loss:1.5f}"
            header_txt += f"\n Epoch {i:3d} losses | {caption}"
    except:
        ### Supports runinfo missing losses by fallback to except
        pass

    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    draw.text((5, 5), header_txt, (255,255,255), font=font)

    # Draw captions
    if losses is not None:
        for i in range(nimages):
            row, col = block_location(i)
            caption = ""
            # if losses is a dict, put all the losses in the caption
            if isinstance(losses, dict):
                for k, v in losses.items():
                    loss = v[i]
                    caption += f" {k}: {loss:1.5f}"
            else:
                caption = f"{losses[i]:1.5f}"
            draw.text((col+pad_width+3, row+height+5), caption, (255,255,255), font=font)

    return img_pil
