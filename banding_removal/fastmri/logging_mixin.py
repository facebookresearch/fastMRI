
import logging
import sys
import pdb
import shutil
from tensorboardX import SummaryWriter


class TensorboardHandler(logging.Handler):
    def __init__(self, writer, tag):
        self.writer = writer
        self.tag = tag
        super().__init__()
    def emit(self, record):
        log_entry = self.format(record)
        #tag, text_string, global_step=None, walltime=None
        if self.writer.file_writer is not None:
            self.writer.add_text(self.tag, log_entry)

class LoggerWriter:
    def __init__(self, level, stream):
        self.level = level
        self._stream = stream 

    def write(self, message):
        if message != '\n':
            self.level(message)
        self._stream.write(message)
        self._stream.flush()

    def flush(self):
        self.level("")
        self._stream.flush()
        
class LoggingMixin(object):
    def initial_setup(self, args):
        #print(f"Setting up logging, rank: {args.rank}")
        root = logging.getLogger()
        root.handlers = []

        if args.debug:
            root.setLevel(logging.DEBUG)
        else:
            root.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s | %(message)s')

        # When using distributed training only send a single process to stdout
        if args.rank == 0:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            root.addHandler(ch)
        
        # send log to a file as well
        fh = logging.FileHandler(self.exp_dir / f'stdout_{args.rank}.log', 'w')
        fh.setFormatter(formatter)
        root.addHandler(fh)

        # For debug messages
        debugh = logging.FileHandler(self.exp_dir / f'debug_{args.rank}.log', 'w')
        debugh.setFormatter(formatter)
        debugh.setLevel(logging.DEBUG)
        root.addHandler(debugh)

        self.tensorboard_dir = self.exp_dir / 'tensorboard'

        if not args.is_distributed:
            shutil.rmtree(str(self.tensorboard_dir), ignore_errors=True)
            self.tensorboard_dir.mkdir(exist_ok=True)

        if args.rank == 0:
            log_dir = self.tensorboard_dir / "main"
        else:
            log_dir = self.tensorboard_dir / f"node{args.rank:03}"
        self.tensorboard = SummaryWriter(log_dir=str(log_dir))
        root.addHandler(TensorboardHandler(self.tensorboard, f"log{args.rank}"))
        logging.info(f"Tensorboard logging to {self.tensorboard_dir.resolve()}")
        self.global_step = 0
        super().initial_setup(args)

    def count_parameters(self, model):
        nparams = 0
        group_idx = 0
        nlayers = 0
        for param in model.parameters():
            group_size = 1
            for g in param.size():
                group_size *= g
            nparams += group_size
            group_idx += 1
            if len(param.shape) >= 2:
                nlayers += 1
        
        return nparams, nlayers

    def model_setup(self, args):
        super().model_setup(args)
        nparams, nlayers = self.count_parameters(self.model)
        logging.info(f"Model parameters: {nparams:,} layers: {nlayers}")

    def start_of_batch_hook(self, progress, logging_epoch):
        super().start_of_batch_hook(progress, logging_epoch)
        self.global_step += 1

    def add_losses_to_tensorboard(self, losses):
        for loss_key, loss_value in losses.items():
            self.tensorboard.add_scalar(loss_key, loss_value, global_step=self.global_step)

    def training_loss_hook(self, progress, losses, logging_epoch):
        super().training_loss_hook(progress, losses, logging_epoch)
        self.add_losses_to_tensorboard(losses)

    def test_loss_hook(self, losses):
        super().test_loss_hook(losses)
        self.add_losses_to_tensorboard(losses)

    def postrun(self):
        logging.info(f"Tensorboard logs at {self.tensorboard_dir.resolve()}")
        if self.args.rank == 0:
            self.tensorboard.export_scalars_to_json(self.exp_dir / "json_tensorboard.json")
        self.tensorboard.close()
