from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
from assignment2 import ncc


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

    height = pyramid[0].size[1]
    # Determine width needed to display pyramid horizontally.
    width = 0
    for img in pyramid:
        width += img.width + 1

    # Create display image for pyramid
    display_pyramid_img = Image.new("L", (width, height), color=255)
    x_offset = 0
    for img in pyramid:
        display_pyramid_img.paste(img, (x_offset, 0))
        x_offset += img.width + 1

    display_pyramid_img.show()


def FindTemplate(pyramid, template, threshold):
    # Matching templates will be drawn on this image.
    display_image = pyramid[0].convert("RGB")

    # Get correlation matrix for image and template
    corr_matrix = ncc.normxcorr2D(pyramid[0], template)

    # Dimensions of display image
    dsp_width, dsp_height = display_image.size
    # Dimensions of template - used for drawing bounding boxes.
    temp_width, temp_height = template.size
    corr_found = 0

    print(f"img shape :{corr_matrix.shape}")
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            # Template matches - draw bounding box.
            if corr_matrix[i, j] > threshold:
                corr_found += 1
                # Points of bounding box (Clipped to display image boundaries.
                top_left = max(0, min(j - temp_width//2, dsp_width-1)), max(0, min(i - temp_height//2, dsp_height-1))

                top_right = max(0, min(j - temp_width//2, dsp_width-1)), max(0, min(i + temp_height//2, dsp_height-1))

                bottom_right = max(0, min(j + temp_width//2, dsp_width-1)), max(0, min(i + temp_height//2, dsp_height-1))

                bottom_left = max(0, min(j + temp_width//2, dsp_width-1)), max(0, min(i - temp_height//2, dsp_height-1))

                draw = ImageDraw.Draw(display_image)
                draw.line((top_left, top_right), fill="red", width=2)
                draw.line((top_right, bottom_right), fill="red", width=2)
                draw.line((bottom_right, bottom_left), fill="red", width=2)
                draw.line((bottom_left, top_left), fill="red", width=2)
                del draw

    print(f"Correlations found: {corr_found}")
    display_image.show()

if __name__ == "__main__":
    # fan = Image.open("faces/fans.jpg")
    # # Get image pyramid
    # pyramid = MakePyramid(fan, 25)
    #
    # # Show pyramid
    # ShowPyramid(pyramid)

    judybats = Image.open("faces/judybats.jpg")
    template = Image.open("faces/template.jpg")
    FindTemplate(MakePyramid(judybats, 50), template, .6)
