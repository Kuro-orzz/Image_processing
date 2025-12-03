import numpy as np
import cv2

def HE_filter(img, alpha):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    HE = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    HE_img = alpha * HE.astype(np.float32) + (1 - alpha) * img.astype(np.float32)
    HE_img = np.clip(HE_img, 0, 255).astype(np.uint8)
    return HE_img


# For img in range [0,1]
def HE_filter_float(img, alpha):
    img8 = (img * 255).astype(np.uint8)

    hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    HE = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    HE = HE.astype(np.float32) / 255.0
    img = img.astype(np.float32)

    HE_img = alpha * HE + (1 - alpha) * img
    HE_img = np.clip(HE_img, 0.0, 1.0)
    return HE_img


# -------- Example usage ----------
if __name__ == "__main__":
    img = cv2.imread("../43.jpg")
    if img is None:
        raise ValueError("Cannot load image.")

    # Example parameters (normally from Micro CNN)
    alpha = 0.8

    HE_img = HE_filter(img, alpha)

    cv2.imshow("Original", img)
    cv2.imshow("Tone Filter", HE_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
