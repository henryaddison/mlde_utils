import logging

from mlde_utils.data import register_action

logger = logging.getLogger(__name__)


@register_action(name="constrain")
class Constrain:
    def __init__(self, query):
        self.query = query

    def __call__(self, ds):
        logger.info(f"Selecting {self.query} portion of dataset")
        return ds.sel(self.query)
