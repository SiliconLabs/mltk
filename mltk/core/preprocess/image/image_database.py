
import collections
import numpy as np

try:
    from cv2 import cv2
except Exception:
    try:
        import cv2
    except:
        raise RuntimeError('Failed import cv2 Python package, try running: pip install opencv-python OR pip install silabs-mltk[full]')



class ImageSignature(list):
    """A list of histograms from the quadrants of an image"""

    def __init__(self, quadrants:int):
        super().__init__()
        self.quadrants = quadrants

    def is_similar(self, other, threshold=0.85) -> bool:
        """Return if the given ImageSignature is similar to this signature
        
        The signatures are considered similar if their histogram correlations
        are with the given threshold
        """
        total_correl = 0.0
        for i, hist in enumerate(self):
            total_correl += abs(cv2.compareHist(hist, other[i], cv2.HISTCMP_CORREL))

        total_correl /= self.quadrants
        return total_correl >= (1.0 - threshold)


class UniqueImageDatabase:
    """Unique image database
    
    Maintain a list of unique image signatures.
    Only unique images are added to the database.
    Similar images are dropped.

    This is useful when generating a dataset to determine 
    when an image should be added or if a similar image is already
    in the database
    """
    
    def __init__(self, maxlen=512, threshold=0.85, quadrants=3):
        """
        Args:
            maxlen: The maximum number of signature to store. 
                The larger this value the more RAM that is required and the longer it takes to add new images to the database
            threshold: The signature comparison threshold. Value closes to 1. indicate more similar images
            quadrants: The number of quadants to divide an image along a the x and y axises
        """
        self._signatures = collections.deque(maxlen=maxlen)
        self._threshold = threshold
        self._quadrants = quadrants


    def add(self, img: np.ndarray) -> bool:
        """Add the image to the database ONLY if it is unique
        
        Args:
            img: The image to potentially add

        Return:
            True if the image is unique and was added
            False if a similar image is already in the database and was not added
        """
        new_sig = create_signature(img, quadrants=self._quadrants)

        # Only add this image if there are no other matching signatures
        for existing_sig in self._signatures:
            if existing_sig.is_similar(new_sig, threshold=self._threshold):
                return False

        # NOTE: We only store up to maxlen signatures
        #       Once this limit is exceeded, we drop old signatures
        self._signatures.append(new_sig)

        return True


    




def create_signature(img: np.ndarray, quadrants=3) -> ImageSignature:
    """Generate a "signature" of the given image by dividing the image
    into quadrants and calculating a historgram of each quadrant.
    The image "signature" is the list of historgrams
    """

    if img.dtype != np.uint8:
        raise ValueError('Image must have a uint8 dtype')

    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[2] == 1:
            img = np.squeeze(img, axis=-1)
        else:
            raise ValueError('Image must have 1 or 3 channels')

    elif len(img.shape) != 2:
        raise ValueError('Image must have shape: height,width or height,width,channels')


    rows, cols = img.shape
    sig = ImageSignature(quadrants=quadrants**2)
    quad_rows = rows // quadrants
    quad_cols = cols // quadrants

    x = 0
    y = 0
    
    while y < rows:
        x = 0
        while x < cols:
            quad = img[y:y+quad_rows, x:x+quad_cols]
            sig.append(cv2.calcHist([quad], [0], None, [8], [0, 256]))
            x += quad_cols
            
        y += quad_rows

    return sig