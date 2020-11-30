"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import time
import sys
from collections import defaultdict
from timeit import default_timer as timer
import datetime
import math
import logging
from collections import OrderedDict, namedtuple
import torch
from torch import autograd
import pdb
from torch.nn import functional as F
import numpy as np
import traceback

from fastmri.data import transforms
from fastmri.common import evaluate
from fastmri.common import utils

class TrainingLoopMixin(object):
    def start_of_batch_hook(self, progress, logging_epoch):
        """ Called at the start of each batch """
        pass

    def midbatch_hook(self, progress, logging_epoch):
        """ Called before backwards(), may be called on sub-batchs """
        pass

    def start_of_test_batch_hook(self, progress, logging_epoch):
        """ Called at the start of each batch """
        pass

    def end_of_test_epoch_hook(self):
        """ Called right at the end of the last test batch """
        pass

    def training_loss_hook(self, progress, losses, logging_epoch):
        """ For logging """
        self.runinfo["epoch"].append(progress)
        self.runinfo["train_losses"].append(losses)

    def test_loss_hook(self, losses):
        """ dev_loss is the primary loss. losses is a dictionary of other losses """
        self.runinfo["dev_losses"].append(losses)

    def predict(self, batch):
        batch = self.preprocess_data(batch)
        prediction = self.model(batch.input)
        return prediction, batch.target

    def training_loss(self, batch):
        output, target = self.predict(batch)
        output, target = transforms.center_crop_to_smallest(output, target)

        if self.args.nan_detection:
            if torch.any(torch.isnan(output)):
                print(output)
                raise Exception("nan encountered")
            if torch.any(torch.isinf(output)):
                print(output)
                raise Exception("inf encountered")

        loss = F.l1_loss(output, target)
        return loss, output, target

    def additional_training_loss_terms(self, loss_dict, batch, prediction, target):
        return loss_dict, batch, prediction, target

    def unnorm(self, output, batch):
        mean, std = batch.mean, batch.std
        return output * std + mean

    def run(self, epoch):
        args = self.args
        self.model.train()
        nbatches = self.nbatches
        interval = timer()
        percent_done = 0
        memory_gb = 0.0
        avg_losses = {}

        if self.args.nan_detection:
            autograd.set_detect_anomaly(True)

        for batch_idx, batch in enumerate(self.train_loader):
            self.batch_idx = batch_idx
            progress = epoch + batch_idx/nbatches

            logging_epoch = (batch_idx % args.log_interval == 0
                             or batch_idx == (nbatches-1))

            self.start_of_batch_hook(progress, logging_epoch)

            if batch_idx == 0:
                logging.info("Starting batch 0")
                sys.stdout.flush()

            def batch_closure(subbatch):
                nonlocal memory_gb

                result = self.training_loss(subbatch)

                if isinstance(result, tuple):
                    result, prediction, target = result
                else:
                    prediction = None # For backwards compatibility
                    target = None

                if isinstance(result, torch.Tensor):
                    # By default self.training_loss() returns a single tensor
                    loss_dict = {'train_loss': result}
                else:
                    # Perceptual loss will return a dict of losses where the main
                    # loss is 'train_loss'. This is for easily logging the parts
                    # composing the loss (eg, perceptual loss + l1)
                    loss_dict = result

                loss_dict, _, _, _ = self.additional_training_loss_terms(
                                        loss_dict, subbatch, prediction, target)

                loss = loss_dict['train_loss']

                # Memory usage is at its maximum right before backprop
                if logging_epoch and self.args.cuda:
                    memory_gb = torch.cuda.memory_allocated()/1000000000

                self.midbatch_hook(progress, logging_epoch)

                self.optimizer.zero_grad()
                self.backwards(loss)
                return loss, loss_dict

            if hasattr(self.optimizer, 'batch_step'):
                loss, loss_dict = self.optimizer.batch_step(batch, batch_closure=batch_closure)
            else:
                closure = lambda: batch_closure(batch)
                loss, loss_dict = self.optimizer.step(closure=closure)

            if args.debug:
                self.check_for_nan(loss)

            # Running average of all losses returned

            for name in loss_dict:
                loss_gpu = loss_dict[name]
                loss_cpu = loss_gpu.cpu().item()
                loss_dict[name] = loss_cpu
                if batch_idx == 0:
                    avg_losses[name] = loss_cpu
                elif batch_idx < 50:
                    avg_losses[name] = (batch_idx*avg_losses[name] + loss_cpu)/(batch_idx+1)
                else:
                    avg_losses[name] = 0.99*avg_losses[name] + 0.01*loss_cpu

            losses = {}
            for name in loss_dict:
                losses['instantaneous_' + name] = loss_dict[name]
                losses['average_' + name] = avg_losses[name]

            self.runinfo['train_fnames'].append(batch['fname'])
            self.training_loss_hook(progress, losses, logging_epoch)

            del losses

            if logging_epoch:
                mid = timer()
                new_percent_done = 100. * batch_idx / nbatches
                percent_change = new_percent_done - percent_done
                percent_done = new_percent_done
                if percent_done > 0:
                    inst_estimate =  math.ceil((mid - interval)/(percent_change/100))
                    inst_estimate = str(datetime.timedelta(seconds=inst_estimate))
                else:
                    inst_estimate = "unknown"

                logging.info('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, inst: {} Mem: {:2.1f}gb'.format(
                    epoch, batch_idx, nbatches,
                    100. * batch_idx / nbatches, loss.item(), inst_estimate,
                    memory_gb))
                interval = mid

                if self.args.break_early is not None and percent_done >= self.args.break_early:
                    break

            if self.args.debug_epoch:
                break

    def stats(self, epoch, loader, setname):
        """ Overriden when doing distributed training """
        losses = self.compute_stats(epoch, loader, setname)
        self.test_loss_hook(losses)
        logging.info(f'Epoch: {epoch}. losses: {losses}')
        return losses["NMSE"]

    #####################################################
    def compute_stats(self, epoch, loader, setname):
        """ This is separate from stats mainly for distributed support"""
        args = self.args
        self.model.eval()
        ndevbatches = len(self.dev_loader)
        logging.info(f"Evaluating {ndevbatches} batches ...")

        recons, gts = defaultdict(list), defaultdict(list)
        acquisition_machine_by_fname = dict()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dev_loader):
                progress = epoch + batch_idx/ndevbatches
                logging_epoch = batch_idx % args.log_interval == 0
                logging_epoch_info = batch_idx % (2 * args.log_interval) == 0
                log = logging.info if logging_epoch_info else logging.debug

                self.start_of_test_batch_hook(progress, logging_epoch)

                batch = self.preprocess_data(batch)
                output, target = self.predict(batch)
                output = self.unnorm(output, batch)
                target = self.unnorm(target, batch)
                fname, slice = batch.fname, batch.slice

                for i in range(output.shape[0]):
                    slice_cpu = slice[i].item()
                    recons[fname[i]].append((slice_cpu, output[i].float().cpu().numpy()))
                    gts[fname[i]].append((slice_cpu, target[i].float().cpu().numpy()))

                    acquisition_type = batch.attrs_dict['acquisition'][i]
                    machine_type = batch.attrs_dict['system'][i]
                    acquisition_machine_by_fname[fname[i]] = machine_type + '_' + acquisition_type

                if logging_epoch or batch_idx == ndevbatches-1:
                    gpu_memory_gb = torch.cuda.memory_allocated()/1000000000
                    host_memory_gb = utils.host_memory_usage_in_gb()
                    log(f"Evaluated {batch_idx+1} of {ndevbatches} (GPU Mem: {gpu_memory_gb:2.3f}gb Host Mem: {gpu_memory_gb:2.3f}gb)")
                    sys.stdout.flush()

                if self.args.debug_epoch_stats:
                    break
                del output, target, batch

            logging.debug(f"Finished evaluating")
            self.end_of_test_epoch_hook()

            recons = {
                fname: np.stack([pred for _, pred in sorted(slice_preds)])
                for fname, slice_preds in recons.items()
            }
            gts = {
                fname: np.stack([pred for _, pred in sorted(slice_preds)])
                for fname, slice_preds in gts.items()
            }

            nmse, psnr, ssims = [], [], []
            ssim_for_acquisition_machine = defaultdict(list)
            recon_keys = list(recons.keys()).copy()
            for fname in recon_keys:
                pred_or, gt_or = recons[fname].squeeze(1), gts[fname].squeeze(1)
                pred, gt = transforms.center_crop_to_smallest(pred_or, gt_or)
                del pred_or, gt_or

                ssim = evaluate.ssim(gt, pred)
                acquisition_machine = acquisition_machine_by_fname[fname]
                ssim_for_acquisition_machine[acquisition_machine].append(ssim)
                ssims.append(ssim)
                nmse.append(evaluate.nmse(gt, pred))
                psnr.append(evaluate.psnr(gt, pred))
                del gt, pred
                del recons[fname], gts[fname]

            if len(nmse) == 0:
               nmse.append(0)
               ssims.append(0)
               psnr.append(0)

            min_vol_ssim = np.argmin(ssims)
            min_vol = str(recon_keys[min_vol_ssim])
            logging.info(f"Min vol ssims: {min_vol}")
            sys.stdout.flush()

            del recons, gts

            acquisition_machine_losses = dict.fromkeys(self.dev_data.system_acquisitions, 0)
            for key, value in ssim_for_acquisition_machine.items():
                acquisition_machine_losses[key] = np.mean(value)

            losses = {'NMSE': np.mean(nmse),
                      'PSNR': np.mean(psnr),
                      'SSIM': np.mean(ssims),
                      'SSIM_var': np.var(ssims),
                      'SSIM_min': np.min(ssims),
                      **acquisition_machine_losses}

        return losses


    def check_for_nan(self, loss):
        def check(x, desc):
            result = torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))
            if result:
                traceback.print_stack(file=sys.stdout)
                print(f"NaN/Inf detected in {desc}")
                pdb.set_trace()
            return result

        check(loss.data, "loss")
        # Check all model parameters, and gradients
        for nm, p in self.model.named_parameters():
            check(p.data, nm)
            if p.grad is not None:
                check(p.grad.data, nm + ".grad")
