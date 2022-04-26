import os 
from typing import List 

from mltk.utils.path import recursive_listdir


def get_supported_build_platforms() -> List[str]:
    """Return a list of supported build platforms"""

    platforms_dir = os.path.dirname(os.path.abspath(__file__)) + '/../shared/platforms'
    retval = []
    for p in recursive_listdir(platforms_dir, regex='.*/CMakeLists.txt'):
        platform_name = os.path.basename(os.path.dirname(p))
        if platform_name == 'common':
            continue
        retval.append(platform_name)
    
    return retval
