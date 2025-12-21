import numpy as np
import cv2

# https://en.wikipedia.org/wiki/Gamma_correction
# γ < 1 is sometimes called an encoding gamma, and the process of encoding with this compressive power-law nonlinearity is called gamma compression.
# γ > 1 is called a decoding gamma, and the application of the expansive power-law nonlinearity is called gamma expansion.
def gamma_correction(img, gamma):
    I_copy = img.copy()
    I_copy = I_copy.astype(np.float32) / 255.0
    I_copy = np.power(I_copy, gamma)
    return (I_copy * 255).clip(0, 255).astype(np.uint8)

if __name__ == '__main__':
    img = cv2.imread("../43.jpg")
    if img is None:
        raise ValueError("Cannot load image.")

    # Example parameters (normally from Micro CNN)
    gamma = 0.8
    gamma_img = gamma_correction(img, gamma)

    cv2.imshow("Original", img)
    cv2.imshow("Tone Filter", gamma_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
