from datetime import datetime
import logging
from pathlib import Path


class Logger:
    def __init__(self, model_name, log_dir, level=logging.INFO, use_tensorboard=False, echo=False):
        self.model_name = model_name
        (Path(log_dir) / "summaries").mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = 'log_{}.txt'.format(date_str)
        formatter = logging.Formatter(
            '[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        self.logger = logging.getLogger("bot")
        # Remove all existing handlers
        self.logger.handlers = []
        # Initialize handlers
        fh = logging.FileHandler(
            Path(log_dir) / Path(log_file))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        if echo:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.tbwriter = None
        if use_tensorboard:
            from tensorboardX import SummaryWriter
            # Tensorboard
            folder = str(
                Path(log_dir) / "summaries" /
                "{}_{}".format(self.model_name, date_str))
            self.tbwriter = SummaryWriter(
                folder
            )

    def info(self, msg, *args):
        self.logger.info(msg, *args)

    def warning(self, msg, *args):
        self.logger.warning(msg, *args)

    def debug(self, msg, *args):
        self.logger.debug(msg, *args)

    def error(self, msg, *args):
        self.logger.error(msg, *args)

    def tb_scalars(self, key, value, step):
        if self.tbwriter is None:
            self.debug("Tensorboard writer is not enabled.")
        else:
            if isinstance(value, dict):
                self.tbwriter.add_scalars(key, value, step)
            else:
                self.tbwriter.add_scalar(key, value, step)
