import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import signal
from scipy.fft import fft2, ifft2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time

# ==============================
# 1. LOAD GAMBAR SENDIRI
# ==============================
img = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Gambar tidak ditemukan!")

clean_img = img.astype(np.float32)

# ==============================
# 2. PSF
# ==============================
def motion_psf(length=15, angle=30):
    psf = np.zeros((length, length))
    center = length // 2

    for i in range(length):
        x = int(center + (i - center) * np.cos(np.deg2rad(angle)))
        y = int(center + (i - center) * np.sin(np.deg2rad(angle)))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1

    psf /= psf.sum()
    return psf

psf = motion_psf(15, 30)

# ==============================
# 3. DEGRADATIONS (SEPERTI PRAKTIKUM)
# ==============================
def apply_motion_blur(image, psf):
    return convolve2d(image, psf, mode='same', boundary='wrap')

def add_gaussian_noise(image, sigma=20):
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 255)

def add_salt_pepper(image, amount=0.05):
    noisy = image.copy()
    num = int(amount * image.size)

    coords = np.random.randint(0, image.size, num)
    noisy.flat[coords] = 255

    coords = np.random.randint(0, image.size, num)
    noisy.flat[coords] = 0

    return noisy

motion_blur_img = apply_motion_blur(clean_img, psf)
gaussian_blur = apply_motion_blur(add_gaussian_noise(clean_img), psf)
sp_blur = apply_motion_blur(add_salt_pepper(clean_img), psf)

degradations = {
    "Motion Blur": (motion_blur_img, psf),
    "Gaussian + Blur": (gaussian_blur, psf),
    "SaltPepper + Blur": (sp_blur, psf)
}

# ==============================
# 4. RESTORASI
# ==============================
def inverse_filter(img, psf, eps=1e-3):
    G = fft2(img)
    H = fft2(psf, s=img.shape)
    return np.abs(ifft2(G / (H + eps)))

def wiener_filter(img, psf, K=0.01):
    G = fft2(img)
    H = fft2(psf, s=img.shape)
    H_conj = np.conj(H)
    return np.abs(ifft2((H_conj / (np.abs(H)**2 + K)) * G))

def lucy_richardson(image, psf, iterations=20):
    estimate = np.full(image.shape, 0.5)
    psf_mirror = psf[::-1, ::-1]

    for _ in range(iterations):
        conv = signal.convolve2d(estimate, psf, 'same')
        relative_blur = image / (conv + 1e-5)
        estimate *= signal.convolve2d(relative_blur, psf_mirror, 'same')

    return estimate

methods = {
    "Inverse": inverse_filter,
    "Wiener": wiener_filter,
    "Lucy": lucy_richardson
}

# ==============================
# 5. PROSES + VISUAL
# ==============================
for name, (degraded, psf) in degradations.items():
    print(f"\n=== {name} ===")

    plt.figure(figsize=(12,6))

    plt.subplot(2,3,1)
    plt.title("Original")
    plt.imshow(clean_img, cmap='gray')

    plt.subplot(2,3,2)
    plt.title(name)
    plt.imshow(degraded, cmap='gray')

    i = 3
    for m_name, method in methods.items():
        start = time.time()
        restored = method(degraded, psf)
        duration = time.time() - start

        mse = np.mean((clean_img - restored)**2)
        p = psnr(clean_img, restored, data_range=255)
        s = ssim(clean_img, restored, data_range=255)

        print(f"{m_name}")
        print(f"MSE   : {mse:.2f}")
        print(f"PSNR  : {p:.2f}")
        print(f"SSIM  : {s:.4f}")
        print(f"Waktu : {duration:.4f} sec\n")

        plt.subplot(2,3,i)
        plt.title(m_name)
        plt.imshow(restored, cmap='gray')
        i += 1

    plt.tight_layout()
    plt.show()

