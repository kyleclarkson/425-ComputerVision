from PIL import Image, ImageDraw
import numpy as np
import csv
import math
import random as rd

def ReadKeys(image):
    """Input an image and its associated SIFT keypoints.

    The argument image is the image file name (without an extension).
    The image is read from the PGM format file image.pgm and the
    keypoints are read from the file image.key.

    ReadKeys returns the following 3 arguments:

    image: the image (in PIL 'RGB' format)

    keypoints: K-by-4 array, in which each row has the 4 values specifying
    a keypoint (row, column, scale, orientation).  The orientation
    is in the range [-PI, PI] radians.

    descriptors: a K-by-128 array, where each row gives a descriptor
    for one of the K keypoints.  The descriptor is a 1D array of 128
    values with unit length.
    """
    im = Image.open(image+'.pgm').convert('RGB')
    keypoints = []
    descriptors = []
    first = True
    with open(image+'.key', 'r') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True)
        descriptor = []
        for row in reader:
            if len(row) == 2:
                assert first, "Invalid keypoint file header."
                assert row[1] == 128, "Invalid keypoint descriptor length in header (should be 128)."
                count = row[0]
                first = False
            if len(row) == 4:
                keypoints.append(np.array(row))
            if len(row) == 20:
                descriptor += row
            if len(row) == 8:
                descriptor += row
                assert len(descriptor) == 128, "Keypoint descriptor length invalid (should be 128)."
                #normalize the key to unit length
                descriptor = np.array(descriptor)
                descriptor = descriptor / math.sqrt(np.sum(np.power(descriptor,2)))
                descriptors.append(descriptor)
                descriptor = []
    assert len(keypoints) == count, "Incorrect total number of keypoints read."
    print("Number of keypoints read:", int(count))
    return [im,keypoints,descriptors]

def AppendImages(im1, im2):
    """Create a new image that appends two images side-by-side.

    The arguments, im1 and im2, are PIL images of type RGB
    """
    im1cols, im1rows = im1.size
    im2cols, im2rows = im2.size
    im3 = Image.new('RGB', (im1cols+im2cols, max(im1rows,im2rows)))
    im3.paste(im1,(0,0))
    im3.paste(im2,(im1cols,0))
    return im3

def DisplayMatches(im1, im2, matched_pairs):
    """Display matches on a new image with the two input images placed side by side.

    Arguments:
     im1           1st image (in PIL 'RGB' format)
     im2           2nd image (in PIL 'RGB' format)
     matched_pairs list of matching keypoints, im1 to im2

    Displays and returns a newly created image (in PIL 'RGB' format)
    """
    im3 = AppendImages(im1,im2)
    offset = im1.size[0]
    draw = ImageDraw.Draw(im3)
    for match in matched_pairs:
        draw.line((match[0][1], match[0][0], offset+match[1][1], match[1][0]),fill="red",width=2)
    im3.show()
    return im3

def match(image1, image2):
    """Input two images and their associated SIFT keypoints.
    Display lines connecting the first 5 keypoints from each image.
    Note: These 5 are not correct matches, just randomly chosen points.

    The arguments image1 and image2 are file names without file extensions.

    Returns the number of matches displayed.

    Example: match('scene','book')
    """
    im1, keypoints1, descriptors1 = ReadKeys(image1)
    im2, keypoints2, descriptors2 = ReadKeys(image2)

    """
    - Each row of descriptor corresponds to vector of feature
    - These vectors are already normalized to unit length.
    """
    # === New code ===
    matched_pairs = []

    # Ratio smallest 2 angles threshold.
    THRESHOLD = .75
    print(f"Threshold: {THRESHOLD}")

    # For each vector in descriptors1, determine angle for each vector in descriptors2
    for index, descriptors_1 in enumerate(descriptors1):
        angles = []
        for descriptors_2 in descriptors2:
            # Compute angle for each vector, add to list
            angles.append(math.acos(np.dot(descriptors_1.T, descriptors_2)))


        # Get smallest and second smallest angles
        min_angle, second_min_angle = sorted(angles)[0:2]
        # If ratio is less than threshold, match keypoints.
        if min_angle / second_min_angle < THRESHOLD:
            # Add matched keypoints to list of matched pairs.
            keypoint_1 = keypoints1[index]
            keypoint_2 = keypoints2[angles.index(min_angle)]
            matched_pairs.append([keypoint_1, keypoint_2])

    print(f"Matches found: {len(matched_pairs)}")

    # == Q 4 ==

    SCALE_THRESHOLD = 0.7
    ANGLE_THRESHOLD = 30 * math.pi / 180 # Convert degrees to radians
    NUM_OF_SAMPLES = 10

    largest_consistent_set = []

    # Sample pair of matched keypoints
    for [keypoint_m_1, keypoint_m_2] in rd.sample(matched_pairs, NUM_OF_SAMPLES):
        '''
            == 4 entries per keypoint: ==
            (row, column, scale, orientation)
        '''
        # Determine differences in scale and orientation
        d_scale_m_1 = keypoint_m_1[2] - keypoint_m_2[2]
        d_angle_m_1 = keypoint_m_1[3] - keypoint_m_2[3] % 2 * math.pi
        consistent_matches = []

        # Check differences against other matches to  determine if consistent.
        for [keypoint_1_m_2, keypoint_2_m_2] in matched_pairs:
            d_scale_m_2 = keypoint_1_m_2[2] - keypoint_2_m_2[2]
            d_angle_m_2 = keypoint_1_m_2[3] - keypoint_2_m_2[3] % 2 * math.pi

            # Compute change in scale and difference of angles.
            dd_scale = abs((d_scale_m_2 / d_scale_m_1) - 1)
            # Ensure difference is min of two possible angles (i.e. at most pi)
            dd_angle = min((d_angle_m_1 - d_scale_m_2) % 2*math.pi, (d_angle_m_2 - d_scale_m_1) % 2*math.pi)

            # Mark matches as consistent if both dd values are less than thresholds
            if dd_scale <= SCALE_THRESHOLD and dd_angle <= ANGLE_THRESHOLD:
                consistent_matches.append([keypoint_1_m_2, keypoint_2_m_2])

        print(f"Consistent matches: {len(consistent_matches)}")

        # Kep largest consistent set
        if len(largest_consistent_set) < len(consistent_matches):
            largest_consistent_set = consistent_matches

    print(f"Largest consistent set size: {len(largest_consistent_set)}")


    # im3 = DisplayMatches(im1, im2, matched_pairs)
    im3 = DisplayMatches(im1, im2, largest_consistent_set)
    im3.save("output.bmp")

    return im3

#Test run...
# match('scene', 'book')
match('library', 'library2')

