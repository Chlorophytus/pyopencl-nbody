import logging


class Loggable:
    """Use this for making N-body contextual objects loggable."""

    def log(self, level: int, string: str) -> None:
        logger = logging.getLogger(type(self).__name__)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            format = logging.Formatter(
                '[%(asctime)s] <%(name)s>\t(%(levelname)s)\t%(message)s')
            handler.setFormatter(format)
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

        logger.log(level, string)
