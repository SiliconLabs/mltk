"""String formatting utilities

See the source code on Github: `mltk/utils/string_formatting.py <https://github.com/siliconlabs/mltk/blob/master/mltk/utils/string_formatting.py>`_
"""

from typing import Union
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
) -> str:
    """Convert the given numeric value to a string with units

    Example:
    0.0314 -> 31.4m

    """
    if value is None:
        return None
    elif value == 0:
        return '0'

    neg_sign = ''
    if value < 0:
        neg_sign = '-'
        value = -value

    if value >= 1e12:
        unit = 'T'
        divisor = 1e12
    elif value >= 1e9:
        unit = 'G'
        divisor = 1e9
    elif value >= 1e6:
        unit = 'M'
        divisor = 1e6
    elif value >= 1e3:
        unit = 'k'
        divisor = 1e3
    elif value >= 1.0:
        unit = ''
        divisor = 1
    elif value >= 1e-3:
        unit = 'm'
        divisor = 1e-3
    elif value >= 1e-6:
        unit = 'u'
        divisor = 1e-6
    elif value >= 1e-9:
        unit = 'n'
        divisor = 1e-9
    else:
        unit = 'p'
        divisor = 1e-12

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