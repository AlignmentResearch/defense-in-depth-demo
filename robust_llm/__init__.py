import logging

import datasets

logger = logging.getLogger(__name__)

# The datasets library has issues with caching and multiple processes.
# To avoid issues, we disable caching of dataset operations.
datasets.disable_caching()
