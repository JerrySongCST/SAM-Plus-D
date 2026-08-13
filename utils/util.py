from scipy import ndimage
import numpy as np
from medpy import metric
import logging
import os
import time
import torch


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def migrate_legacy_state_dict(sd):
    """Remap keys from pre-rename checkpoints (MoLoRA / TSM era) to current names.

    The released Google-Drive checkpoints were saved before we renamed the
    modules to DRLoRA / DSM. Their state_dict keys still reference the old
    attribute names (``molora_q``, ``molora_v``). This helper rewrites those
    keys so ``model.load_state_dict(...)`` succeeds on the current model.

    New keys are strictly additive replacements; no tensor values change.
    """
    remaps = [(".molora_q.", ".drlora_q."), (".molora_v.", ".drlora_v.")]
    out = {}
    for k, v in sd.items():
        nk = k
        for old, new in remaps:
            nk = nk.replace(old, new)
        out[nk] = v
    return out
