import numpy as np
import cv2

def white_balance(img, Wb, Wg, Wr):
    I_copy = img.copy()
    I_copy[:, :, 0] *= Wb
    I_copy[:, :, 1] *= Wg
    I_copy[:, :, 2] *= Wr
    return np.clip(I_copy, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    img = cv2.imread('../43.jpg')

    # wb = cv2.xphoto.createGrayworldWB()
    # balanced_img = wb.balanceWhite(img)

    Wb, Wg, Wr = 1.1, 1.05, 1.0
    Wb = white_balance(img, Wb, Wg, Wr)

    cv2.imshow("Original", img)
    cv2.imshow("Tone Filter", Wb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
