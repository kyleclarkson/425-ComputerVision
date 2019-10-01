from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc


def MakePyramid(image, minsize):
    '''
    :param image: PIL input image
    :param minsize: a positive value such that no down-sampled image in the pyramid
            has a dimension of smaller size.
    :return: An image pyramid as a list, starting with the input image
            and then consecutive downscaling.
    '''

    assert isinstance(image, Image.Image), "Image must be Pillow image!"
    assert minsize > 0, "minsize must be positive!"

    # List to return
    return_list = []
    # Get image size
    width, height = image.size

    while width >= minsize and height >= minsize:
        return_list.append(image)
        image = image.resize((int(width*0.75), int(height*0.75)))
        width, height = image.size

    return return_list


def ShowPyramid(pyramid):
    '''
    :param pyramid: The pyramid to be displayed
    '''
    # Determine width needed to display pyramid horizontally.
    width = 0
    for img in pyramid:
        width += img.width + 1

    # Create display image for pyramid
    display_pyramid_img = Image.new("L", (width, fan.height), color=255)
    x_offset = 0
    for img in pyramid:
        display_pyramid_img.paste(img, (x_offset, 0))
        x_offset += img.width + 1

    display_pyramid_img.show()


if __name__ == "__main__":
    fan = Image.open("faces/fans.jpg")
    # Get image pyramid
    pyramid = MakePyramid(fan, 25)

    # Show pyramid
    ShowPyramid(pyramid)
