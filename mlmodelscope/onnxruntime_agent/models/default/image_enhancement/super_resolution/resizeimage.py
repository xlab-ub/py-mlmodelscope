from __future__ import division
import math
# import sys
from functools import wraps

from PIL import Image

class ImageSizeError(Exception):
    """
    Raised when the supplied image does not
    fit the intial size requirements
    """
    def __init__(self, actual_size, required_size):
        self.message = 'Image is too small, Image size : %s, Required size : %s' % (actual_size, required_size)
        self.actual_size = actual_size
        self.required_size = required_size

    def __str__(self):
        return repr(self.message)

def validate(validator):
    """
    Return a decorator that validates arguments with provided `validator`
    function.

    This will also store the validator function as `func.validate`.
    The decorator returned by this function, can bypass the validator
    if `validate=False` is passed as argument otherwise the fucntion is
    called directly.

    The validator must raise an exception, if the function can not
    be called.
    """

    def decorator(func):
        """Bound decorator to a particular validator function"""

        @wraps(func)
        def wrapper(image, size, validate=True, *args, **kwargs):
            if validate:
                validator(image, size)
            return func(image, size, *args, **kwargs)
        return wrapper

    return decorator


def _is_big_enough(image, size):
    """Check that the image's size superior to `size`"""
    if (size[0] > image.size[0]) and (size[1] > image.size[1]):
        raise ImageSizeError(image.size, size)

@validate(_is_big_enough)
def resize_crop(image, size):
    """
    Crop the image with a centered rectangle of the specified size
    image:      a Pillow image instance
    size:       a list of two integers [width, height]
    """
    img_format = image.format
    image = image.copy()
    old_size = image.size
    left = (old_size[0] - size[0]) / 2
    top = (old_size[1] - size[1]) / 2
    right = old_size[0] - left
    bottom = old_size[1] - top
    rect = [int(math.ceil(x)) for x in (left, top, right, bottom)]
    left, top, right, bottom = rect
    crop = image.crop((left, top, right, bottom))
    crop.format = img_format
    return crop

@validate(_is_big_enough)
def resize_cover(image, size, resample=Image.LANCZOS):
    """
    Resize image according to size.
    image:      a Pillow image instance
    size:       a list of two integers [width, height]
    """
    img_format = image.format
    img = image.copy()
    img_size = img.size
    ratio = max(size[0] / img_size[0], size[1] / img_size[1])
    new_size = [
        int(math.ceil(img_size[0] * ratio)),
        int(math.ceil(img_size[1] * ratio))
    ]
    img = img.resize((new_size[0], new_size[1]), resample)
    img = resize_crop(img, size)
    img.format = img_format
    return img
