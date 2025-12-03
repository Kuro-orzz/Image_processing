# source: https://doi.org/10.5201/ipol.2024.530

import numpy as np
import cv2
import sys

# I: base [0, 1]
# sz: Size of patch
# return base [0, 1]
def DarkChannel(I, sz=15):
    b, g, r = cv2.split(I)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark_channel = cv2.erode(dc, kernel)
    return dark_channel

# Estimate A
# I: base [0, 1]
# dark_channel: base [0, 1]
# p: percentage of pixel in I use to estimate A
# return base [0, 1]
def Ambient_Light_Estimate(I, dark_channel, p=1e-3):
    h, w = dark_channel.shape
    N = h * w
    # Flat and sort dark channel
    flat_dark = dark_channel.reshape(N)
    sorted_dark = flat_dark.argsort()
    sorted_dark = sorted_dark[::-1]
    flat_I = I.reshape(N, 3)       # 3D->2D

    # Number of pixel used for the estimation
    M = max(1, int(p * N))    
    M = min(M, N)

    A = np.zeros([1, 3], dtype=np.float32)
    for i in range(M):
        A += flat_I[sorted_dark[i]]            # Add pixel color
    A /= M
    return A

# Estimate t(x)
# I: base [0, 1]
# A: base [0, 1]
def Transmission_Map_Estimate(I, A, sz, w=0.95):
    copy_I = np.empty(I.shape, I.dtype)
    for c in range(3):
        copy_I[:, :, c] = I[:, :, c] / A[0, c]
    t_x = 1 - w * DarkChannel(copy_I, sz)
    return t_x

# I is one channel of image and base [0, 1]
# accumulated values of image (2D prefix sum)
# Note: faster way is just return np.cumsum(np.cumsum(I, axis=0), axis=1)
def Integral_Image(I):
    w, h = I.shape
    II = np.zeros(I.shape, dtype=np.float32)
    II[0, 0] = I[0, 0]
    for x in range(1, w):
        II[x, 0] = II[x-1, 0] + I[x, 0]
    for y in range(1, h):
        s = I[0, y]
        II[0, y] = II[0, y-1] + s
        for x in range(1, w):
            s += I[x, y]
            II[x, y] = II[x, y-1] + s
    return II

# only apply for one channel
def BoundaryHandling(I, rr):
    return np.pad(I, ((rr, rr), (rr, rr)), mode='reflect')

# I is one channel of img and base [0, 1]
# rr is radius of patch (r = 2rr + 1, patch rxr)
# Note: faster way is just return cv2.boxFilter(I,cv2.CV_64F,(rr,rr))
def Patch_Averages(I, rr):
    w, h = I.shape  # row, col (swap for purpose)
    I_avg = np.zeros(I.shape, dtype=np.float32)
    I_0 = BoundaryHandling(I, rr)
    II_0 = Integral_Image(I_0)
    for y in range(h):
        for x in range(w):
            # Coordinate of (x, y) in integral img
            x_0 = x + rr
            y_0 = y + rr

            # Get prefix sum 2D
            I_avg[x, y] = II_0[x_0 + rr, y_0 + rr]
            if x_0 - rr - 1 >= 0:
                I_avg[x, y] -= II_0[x_0 - rr - 1, y_0 + rr]
            if y_0 - rr - 1 >= 0:
                I_avg[x, y] -= II_0[x_0 + rr, y_0 - rr - 1]
            if x_0 - rr - 1 >= 0 and y_0 - rr - 1 >= 0:
                I_avg[x, y] += II_0[x_0 - rr - 1, y_0 - rr - 1]
            I_avg[x, y] /= (2 * rr + 1) ** 2
    return I_avg

# I: base [0, 1]
# return float base [0, 1]
def Guided_Filter(I, t, rr, eps):
    mean_I = Patch_Averages(I, rr)
    mean_I2 = Patch_Averages(I*I, rr)
    var_I = mean_I2 - mean_I * mean_I
    mean_t = Patch_Averages(t, rr)
    mean_It = Patch_Averages(I * t, rr)
    covar_It = mean_It - mean_I * mean_t
    a = covar_It / (var_I + eps)
    b = mean_t - a * mean_I
    mean_a = Patch_Averages(a, rr)
    mean_b = Patch_Averages(b, rr)
    t_0 = mean_a * I + mean_b
    return t_0

# I: base [0, 1]
# return base [0, 1]
def TransmissionRefine(I, t, r=61, eps=1e-3):
    I_gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY);
    t = Guided_Filter(I_gray, t, r, eps);
    t = np.clip(t, 0, 1)
    return t

# I: base [0, 1]
# return base [0, 1]
def Recover_Scene_Radiance(I, t, A, t0=0.1):
    I_dehaze = np.zeros_like(I, dtype=float)
    for c in range(3):
        I_dehaze[:, :, c] = (I[:, :, c] - A[0, c]) / np.maximum(t, t0) + A[0, c]
    I_dehaze = np.clip(I_dehaze, 0, 1)
    return I_dehaze

# Using Dark Channel Prior (DCP) Algorithm
# I: base [0, 255]
# return base [0, 1]
def Dehazing(I, sz=15, p=1e-3, w=0.95, r=61, eps=0.0001, t0=0.1):
    I = I.astype(np.float32) / 255.0        # Normalize to base [0, 1]
    J_dark = DarkChannel(I, sz)
    A = Ambient_Light_Estimate(I, J_dark, p)
    t = Transmission_Map_Estimate(I, A, sz, w)
    t_refine = TransmissionRefine(I, t, r, eps)
    J = Recover_Scene_Radiance(I, t_refine, A, t0)
    return J

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
    else:
        img = cv2.imread('fog.jpg')
    if img is None:
        print("error")
        exit()
    print(img.shape)
    dehazed = Dehazing(img)
    cv2.imwrite('res.jpg', (dehazed*255).astype(np.uint8))
    cv2.imshow('Dehazed', dehazed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()