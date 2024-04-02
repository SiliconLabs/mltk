"""String formatting utilities

See the source code on Github: `mltk/utils/string_formatting.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/string_formatting.py>`_
"""

from typing import Union
from collections import OrderedDict
import datetime


class FormattedInt(int):
    def __str__(self):
        return format_units(int(self))

class FormattedFloat(float):
    def __str__(self):
        return format_units(float(self))


def format_units(
    value: Union[int,float],
    precision:int=3,
    add_space:bool=True,
    ljust:int=0,
    rjust:int=0,
    memory_units:bool=False
) -> str:
    """Convert the given numeric value to a string with units

    Example:
    0.0314 -> 31.4m

    """
    if value is None:
        return None
    elif value == 0:
        return '0'
    
    if '_BASE_10_UNITS_TABLE' not in globals():
        _BASE_10_UNITS_TABLE = OrderedDict()
        _BASE_10_UNITS_TABLE['T'] = 1e12
        _BASE_10_UNITS_TABLE['G'] = 1e9
        _BASE_10_UNITS_TABLE['M'] = 1e6
        _BASE_10_UNITS_TABLE['k'] = 1e3
        _BASE_10_UNITS_TABLE[''] = 1
        _BASE_10_UNITS_TABLE['m'] = 1e-3
        _BASE_10_UNITS_TABLE['u'] = 1e-6
        _BASE_10_UNITS_TABLE['n'] = 1e-9
        _BASE_10_UNITS_TABLE['p'] = 1e-12
        globals()['_BASE_10_UNITS_TABLE'] = _BASE_10_UNITS_TABLE
    if '_BASE_2_UNITS_TABLE' not in globals():
        _BASE_2_UNITS_TABLE = OrderedDict()
        _BASE_2_UNITS_TABLE['T'] = 1024 ** 4
        _BASE_2_UNITS_TABLE['G'] = 1024 ** 3
        _BASE_2_UNITS_TABLE['M'] = 1024 ** 2
        _BASE_2_UNITS_TABLE['k'] = 1024
        _BASE_2_UNITS_TABLE[''] = 1
        globals()['_BASE_2_UNITS_TABLE'] = _BASE_2_UNITS_TABLE

    units_table = globals().get('_BASE_2_UNITS_TABLE' if memory_units else '_BASE_10_UNITS_TABLE')

    neg_sign = ''
    if value < 0:
        neg_sign = '-'
        value = -value

    unit = ''
    divisor = 1
    for unit, divisor in units_table.items():
        if value >= divisor:
            break

    space_str = ' ' if add_space else ''

    if divisor == 1:
        fmt = '{:.%df}' % precision
        retval = fmt.format(value)

    elif precision > 0:
        fmt = '{:.%df}' % precision
        retval = fmt.format(value / divisor) + space_str + unit

    else:
        retval = f'{int(value / divisor)}{space_str}{unit}'

    retval = neg_sign + retval

    if ljust > 0:
        retval = retval.ljust(ljust)
    if rjust > 0:
        rjust = retval.rjust(rjust)
    return retval


def convert_units(v:str) -> int:
    """Convert human-readable string representing memory size to 
    integer corresponding to memory size in bytes
    
    e.g.:
        '32kb'   -> 32768
        '2*128M' -> 268435456
    """

    multiplier_map = dict(
        kilobyte    = 1024,
        megabyte    = 1024 ** 2,
        gigabyte    = 1024 ** 3,
        terabyte    = 1024 ** 4,
        petabyte    = 1024 ** 5,
        exabyte     = 1024 ** 6,
        zetabyte    = 1024 ** 7,
        yottabyte   = 1024 ** 8,
        kb          = 1024,
        k           = 1024,
        mb          = 1024**2,
        m           = 1024**2,
        gb          = 1024**3,
        g           = 1024**3,
        tb          = 1024**4,
        t           = 1024**4,
        pb          = 1024**5,
        p           = 1024**5,
        eb          = 1024**6,
        e           = 1024**6,
        zb          = 1024**7,
        z           = 1024**7,
        yb          = 1024**8,
        y           = 1024**8,
        byte        = 1,
        b           = 1
    )

    if v is None:
        return None 
    
    if isinstance(v, (int,float)):
        return int(v)

    if not isinstance(v, str):
        raise ValueError('Invalid argument')

    x = v.lower().strip().strip('s')

    for suffix, multiplier in multiplier_map.items():
        if x.endswith(suffix):
            x = x[:-len(suffix)].strip()
            x = eval(x) # pylint: disable=eval-used
            x = int(float(x) * multiplier)
            return x

    if x.startswith('0x'):
        x = x[2:]
        base = 16
    else:
        base = 10

    return int(x, base)


def pretty_time_str() -> str:
    """Return the current time as Y-m-d H-M-S """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def iso_time_str() -> str:
    """Return the current time as ISO 8601 format
    e.g.: 2019-01-19T23:20:25.459Z
    """
    now = datetime.datetime.utcnow()
    return now.isoformat()[:-3]+'Z'


def iso_time_filename_str() -> str:
    """Return the current time as ISO 8601 format
    that is suitable for a filename
    e.g.: 2019-01-19T23-20-25-459
    """
    now = datetime.datetime.utcnow()
    return now.isoformat()[:-3].replace(':', '-')