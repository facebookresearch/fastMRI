import torch
import torchvision
import logging
import sys
import pdb
import numpy as np
from torch.nn import functional as F
from .common import image_grid
from .data import transforms
from fastmri.common import evaluate
from PIL import Image

class VisualizationMixin(object):
    def initial_setup(self, args):
        super().initial_setup(args)
        self.example_idx = 3 # From display loader

    def start_of_epoch_hook(self, epoch):
        if epoch == 0 and self.args.visual_first_epoch:
            self.visualize_dev(epoch)
            if not self.args.is_distributed:
                self.visualize_data_transform()

        super().start_of_epoch_hook(epoch)

    def end_of_epoch_hook(self, epoch):
        self.visualize_dev(epoch+1) #Start indexing at 1

        super().end_of_epoch_hook(epoch)

    def quantiles(self, x):
        x_np = x.flatten().to('cpu').numpy()
        qs = np.quantile(x_np, (0.001, 0.01, 0.1, 0.9, 0.99, 0.999))
        return ", ".join([f"{q:+1.1e}" for q in qs])

    def visualize_dev(self, epoch):
        self.model.eval()

        grid_size = self.args.display_count
        if grid_size == 0:
            return

        logging.debug("Saving visualizations")
        images_processed = 0

        grid_recons = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.display_loader):
                output, target = self.predict(batch)
                target = transforms.center_crop_or_pad(target,
                        (self.args.resolution_height, self.args.resolution_width))
                output = transforms.center_crop_or_pad(output,
                        (self.args.resolution_height, self.args.resolution_width))

                if batch_idx == self.example_idx:
                    logging.info(f"output std: {output.std()} target std {target.std()}")
                    logging.info(f"output min: {output.min()} target min {target.min()}")
                    logging.info(f"output max: {output.max()} target max {target.max()}")
                    logging.debug(f"output (0.1, 1, 10, 90, 99, 99.9): {self.quantiles(output)}")
                    logging.debug(f"target (0.1, 1, 10, 90, 99, 99.9): {self.quantiles(target)}")

                if grid_recons is None:
                    grid_recons = torch.zeros(grid_size, output.shape[1],
                        output.shape[2], output.shape[3]).to(self.device)
                    grid_images = torch.zeros_like(grid_recons)
                    grid_iffts = torch.zeros_like(grid_recons)

                for j in range(output.shape[0]):
                    if images_processed >= grid_size:
                        break
                    grid_recons[images_processed, ...] = output.data[j, ...].float()
                    grid_images[images_processed, ...] = target.data[j, ...].float()

                    if self.args.display_ifft:
                        masked_kspace = batch['input']
                        ifft_abs = transforms.complex_abs(transforms.ifft2(masked_kspace)).squeeze(0)
                        masked_image = transforms.root_sum_of_squares(ifft_abs).unsqueeze(0)
                        masked_image = transforms.center_crop_or_pad(masked_image,
                                (self.args.resolution_height, self.args.resolution_width))
                        grid_iffts[images_processed, ...] = masked_image.data[j, ...].float()

                    images_processed += 1

            logging.debug(f"Copying visual images to cpu")
            sys.stdout.flush()
            grid_recons = grid_recons.cpu()
            grid_images = grid_images.cpu()
            grid_errors = torch.abs(grid_recons - grid_images)

            if self.args.rank == 0: # Only master task does visual
                self.save_images(grid_images, 'Target', epoch)
                self.save_images(grid_recons, 'Reconstruction', epoch)
                self.save_images(grid_errors, 'Error', epoch)

            logging.debug(f"Sent images to tensorboard and saved.")
            sys.stdout.flush()

            if self.args.display_ifft and self.args.rank == 0:
                grid_iffts = grid_iffts.cpu()
                self.save_images(grid_iffts, 'Ifft', epoch)

            image_dir = self.exp_dir / "grids"
            image_dir.mkdir(exist_ok=True)

            image_blocks = []
            losses = {'NMSE': [], 'SSIM': [], 'MSE': []}
            for i in range(images_processed):
                gtnp = grid_images[i].cpu().numpy()
                prednp = grid_recons[i].cpu().numpy()
                losses['NMSE'].append(evaluate.nmse(gtnp, prednp))
                losses['SSIM'].append(evaluate.ssim(gtnp, prednp))
                losses['MSE'].append(evaluate.mse(gtnp, prednp))

                gt = grid_images[i]
                shift = torch.min(gt)
                scale = torch.max(gt - shift)

                image_blocks.append((
                    (grid_images[i] - shift) / scale,
                    (grid_recons[i] - shift) / scale,
                    0.5 + 4 * (grid_errors[i] / scale)) +
                    (((grid_iffts[i] - shift) / scale, ) if self.args.display_ifft else ())
                    )

            grid_pil = image_grid.grid(image_blocks, losses=losses, runinfo=self.runinfo)
            grid_path = image_dir / f"epoch{epoch:03}.png"
            if self.args.rank == 0:
                grid_pil.save(grid_path, format="PNG")
            logging.info(f"Saved image grid to {grid_path.resolve()}")
            sys.stdout.flush()

    def save_images(self, image, tag, epoch):
        image = image.float()
        image = image - image.min()
        image = image / image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        self.tensorboard.add_image(tag, grid, epoch)

    def visualize_data_transform(self):
        """
            Just save to disk the raw input data for an example instance.
            The two complex numbers are shown side-by-side, and the channel
            dimension is stacked vertically in order.
        """
        logging.debug("Saving out raw input grid")
        sys.stdout.flush()

        with torch.no_grad():

            for bidx, batch in enumerate(self.display_loader):
                if bidx == self.example_idx:
                    break
            input, *_ = self.preprocess_data(batch)

            # If the input has a slice dimension, take the first slice
            if input.dim() == 6:
                input = input[:, 0, ...]

            # Only plot input data that is complex valued
            if input.shape[-1] != 2:
                return

            req_shape = (input.shape[1]*2, 1, input.shape[2], input.shape[3])
            channels = input[0, :, None, ...].transpose(1, 4).contiguous().view(req_shape)
            grid = torchvision.utils.make_grid(channels, nrow=2, padding=2, normalize=True)
            grid_np = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            grid_pil = Image.fromarray(grid_np)

            grid_path = self.exp_dir / "grids" / f"raw_grid.png"
            if self.args.rank == 0:
                grid_pil.save(grid_path, format="PNG")
            logging.info(f"Saved raw input grid to {grid_path.resolve()}")
            sys.stdout.flush()
