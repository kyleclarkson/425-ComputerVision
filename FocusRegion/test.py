import numpy as np
import cv2
# from skimage.measure import structural_similarity as ssim


def mse(img1, img2):
    """ Compute mean square error of two images. """
    assert img1.shape == img2.shape

    err = np.sum((img1.astype('float') - img2.astype("float")) ** 2 )
    return err / float(img1.shape[0] * img1.shape[1])


def query_region(img1, img2, x1, x2, y1, y2):
    """ Replace subimage centered at center of img1 with
     corresponding the subimage from img2. Return new image. """
    result = img1.copy()
    result[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    return result


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    isCapturing = True
    while isCapturing:
        # capture consecutive frames.
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()

        frame_height = frame1.shape[0]
        frame_width = frame1.shape[1]

        # Convert to gray scale.
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


        """
        TODO
        - Capture two consecutive images img1, img2. 
        - Allow agent to choose k (hyper.param) centers of various sizes (model learns)
            to update img1 with corresponding subimages from img2. Quality of choice is 
            determined by an error (e.g. mse between img2 and updated img1.)
        - Allow training on various videos.
        - (?) Different way to represent image than raw image? 
            Dimensional reduction? 
        """
        # Display the resulting frame
        # cv2.imshow('frame', diff)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            isCapturing = False

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


