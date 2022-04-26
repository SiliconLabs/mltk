from typing import Union



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



