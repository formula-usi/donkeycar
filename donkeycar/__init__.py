import os
import sys
from pyfiglet import Figlet
import logging

__version__ = '5.1.0'

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

f = Figlet(font='speed')


print(f.renderText('Donkey Car'))
print(f'using donkey v{__version__} ...')

if sys.version_info < (3, 10):
    msg = f'Donkey requires Python 3.10 or greater. You are using {sys.version}'
    raise ValueError(msg)

# The default recursion limits in CPython are too small.
sys.setrecursionlimit(10**5)

from .vehicle import Vehicle
from .memory import Memory
from . import utils
from . import config
from . import contrib
from .config import load_config
