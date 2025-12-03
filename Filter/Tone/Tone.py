import numpy as np
import cv2

def tone_filter(img, t_params):
    # Normalize image to [0,1]
    I_copy = img.astype(np.float32) / 255.0

    # prefix Tk = cumulative sum t0..t(k-1)
    T_prefix = np.cumsum(t_params)
    T8 = T_prefix[-1]  # total sum

    # I_tone in range [0,1]
    I_tone = np.zeros_like(I_copy)

    sigma = np.zeros_like(I_tone)
    for k in range(8):
        value = np.clip(8 * I_copy - k, 0, 1) * T_prefix[k]
        sigma += value
    I_tone = sigma / T8
    return (I_tone * 255).clip(0, 255).astype(np.uint8)


# -------- Example usage ----------
if __name__ == "__main__":
    img = cv2.imread("../43.jpg")
    if img is None:
        raise ValueError("Cannot load image.")

    # Example t parameters (normally from Micro CNN)
    t_params = [0.9, 0.7, 1.0, 1.2, 0.8, 1.1, 0.6, 1.0]

    tone_img = tone_filter(img, t_params)

    cv2.imshow("Original", img)
    cv2.imshow("Tone Filter", tone_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
