import numpy as np
from datetime import date

EXBI = 2**60
EIB = EXBI  # a convenience alias
EXA = 10**18
PIB = 2**50
GIB = 2**30
SECTOR_SIZE = 32 * GIB

PIB_PER_SECTOR = SECTOR_SIZE / PIB
EIB_PER_SECTOR = SECTOR_SIZE / EIB

NETWORK_START = np.datetime64(date(2020, 10, 15))
