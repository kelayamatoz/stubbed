from importlib.abc import SourceLoader, MetaPathFinder
from importlib.util import spec_from_file_location
from importlib.machinery import FileFinder, SourceFileLoader

import importlib

import os
import sys

class HookedLoader(SourceFileLoader):
    def __init__(self, fullname, path):
        super().__init__(fullname, path)
        self.fullname = fullname
        self.path = path

    def get_data(self, path):
        print("Get Data:", path)
        return super().get_data(path)


def install():
    loader_details = HookedLoader, [".py"]
    # insert the path hook ahead of other path hooks
    sys.path_hooks.insert(0, FileFinder.path_hook(loader_details))
    # clear any loaders that might already be in use by the FileFinder
    sys.path_importer_cache.clear()
    importlib.invalidate_caches()

install()