import torch
import numpy as np
from torch.utils.data import Sampler
import torch.distributed as dist


class VolumeSampler(Sampler):
    """
        Based on pytorch DistributedSampler, the difference is that all instances from the same
        volume need to go to the same node. Dataset example is a list of tuples (fname, instance),
        where fname is essentially the volume name (actually a filename).
    """

    def __init__(self, dataset):
        """
        Args:
            dataset: Dataset used for sampling.
        """
        self.dataset = dataset
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.epoch = 0

        # All nodes
        self.all_volume_names = np.array(sorted(list({example[0] for example in self.dataset.examples})))
        self.all_volumes_split = np.array_split(self.all_volume_names, self.world_size)

        # This node
        self.volumes = self.all_volumes_split[self.rank]
        self.indices = []

        for i, example in enumerate(self.dataset.examples):
            vname = example[0]
            if vname in self.volumes:
                self.indices.append(i)


        self.total_size = len(self.dataset.examples)
        self.indices = np.array(self.indices)
        self.num_samples = len(self.indices)

    def __iter__(self):
        # Deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        ordering = torch.randperm(self.num_samples, generator=g).tolist()
        indices_shuffled = self.indices[ordering]
        return iter(indices_shuffled.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
