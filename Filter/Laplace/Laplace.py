import cv2
import numpy as np

def Laplace_filter(img, lambda_var):
    I_copy = img.copy().astype(np.float32)

    # Compute Laplacian per channel
    lap_op = cv2.Laplacian(I_copy, cv2.CV_32F, ksize=3)

    # Condition masks
    mask_pos = lap_op > 0
    mask_neg = lap_op < 0

    I_L = I_copy.copy()
    I_L[mask_neg] = I_copy[mask_neg] + lambda_var * (I_copy[mask_neg] - lap_op[mask_neg])
    I_L[mask_pos] = I_copy[mask_pos] + lambda_var * (I_copy[mask_pos] + lap_op[mask_pos])
    
    return np.clip(I_L, 0, 255).astype(np.uint8)


# -------- Example usage ----------
if __name__ == "__main__":
    img = cv2.imread("../43.jpg")
    if img is None:
        raise ValueError("Cannot load image.")

    # Example parameters (normally from Micro CNN)
    alpha = 0.8

    Laplace = Laplace_filter(img, alpha)

    cv2.imshow("Original", img)
    cv2.imshow("Tone Filter", Laplace)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
