import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # capture consecutive frames.
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    # Convert to gray scale, normalize to [0,1]
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    img1 = np.zeros(gray1.shape)
    img2 = np.zeros(gray2.shape)

    img1 = cv2.normalize(gray1, img1, 0, 1, cv2.NORM_MINMAX)
    img2 = cv2.normalize(gray2, img2, 0, 1, cv2.NORM_MINMAX)

    # TODO take difference, clip
    diff = img1-img2
    cv2.imshow('frame', diff)
    # break
    # Display the resulting frame
    # cv2.imshow('frame', gray1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()