"""
Logging configuration, called once by the entry point.
"""

import logging
import sys
from pathlib import Path

from pi05.distributed import DistInfo


def init_logging(info: DistInfo, log_file: Path | None = None) -> None:
    """
    Rank 0 logs INFO and above (terminal + log_file); other ranks only WARNING and above.
    """
    
    level = logging.INFO if info.is_main else logging.WARNING
    formatter = logging.Formatter(
        fmt=f"%(asctime)s %(levelname)s [rank {info.rank}] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None and info.is_main:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))  # appends: a resumed run continues it

    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=handlers, force=True)