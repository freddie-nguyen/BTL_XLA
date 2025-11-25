from PIL import Image
import numpy as np
from scipy import ndimage
import cv2


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    '''
    Chuyển ảnh rgb sang ảnh xám

    :param rgb: ảnh màu RGB; type np.ndarray
    :return: ảnh xám
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # using luminosity method for converting rgb to gray level
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def gaussian_kernel(kernel_size: int, sigma=1) -> np.ndarray:
    '''

    :param kernel_size: kích thước của cửa sổ
    :param sigma: độ lệch chuẩn trong phân phối gaussian
    :return:
    '''
    size = int(kernel_size) // 2
    '''
    tạo lưới tọa độ
    x = [[-2 -2 -2 -2 -2]
         [-1 -1 -1 -1 -1]
         [ 0  0  0  0  0]
         [ 1  1  1  1  1]
         [ 2  2  2  2  2]] 
    y = [[-2 -1  0  1  2]
         [-2 -1  0  1  2]
         [-2 -1  0  1  2]
         [-2 -1  0  1  2]
         [-2 -1  0  1  2]]
    '''
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    # tạo bộ lọc gaussian
    g = (np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
         * (1 / (2.0 * np.pi * sigma ** 2)))
    return g


def sobel_filters(img):
    """
    Tính xấp xỉ gradient theo toán tử sobel
    Sobel:
        df/dx
            [-1 -2 -1
              0  0  0
              1  2  1]
        df/dy
            [-1  0 -1
             -2  0 -2
             -1  0 -1]
    Gradient Magnitude và Direction bằng Sobel Operators.
    """
    Kx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ky = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)  # ma trận đạo hàm theo x
    Iy = ndimage.filters.convolve(img, Ky)  # ma trận đạo hàm theo y

    G = np.hypot(Ix, Iy)  # -> magnitude of gradient vector: sqrt(Ix ** 2 + Iy ** 2)
    G = G / G.max() * 255  # scaling to 0 -> 255
    theta = np.arctan2(Iy, Ix)  # ma trận theo hướng của gradiant
    return (G, theta)


def edge_thinning(img, D):
    """
    Non-Maximum Suppression (Lower Bound Thresholding).
    loại bỏ những điểm không đạt giá trị cực đại
    """
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    # chuyển đổi ma trận hướng gradient từ radian sang độ
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]

                # nếu pixel có giá trị lớn hơn cả 2 lân cận theo hướng vector
                # thì nó là cực đại, tức pixel biên
                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0
            except IndexError:
                pass
    return Z


def dbl_threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    '''

    phát hiện biên bằng ngưỡng kép
    T1 ~ lowThreshold
    T2 ~ highThreshold

    :param img:
    :param lowThresholdRatio:
    :param highThresholdRatio:
    :return:
    '''
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255)

    # trả về tọa độ theo x (i) và tọa độ theo y (j) của các điểm biên mạnh
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    # put strong=int32(255) giá trị điểm biên mạnh, idk
    # put weak=int32(255)   giá trị điểm biên yếu
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    '''
    Mục đích của bước này là biến các Biên Yếu (Weak Edges)
    có liên kết với Biên Mạnh (Strong Edges) thành biên Mạnh,
    đồng thời loại bỏ các pixel Yếu không liên kết (thường là nhiễu).
    Điều này giúp tạo ra các đường biên liên tục và giảm thiểu nhiễu biên giả.

    :param img:
    :param weak: value of weak edge pixel
    :param strong:
    :return:
    '''
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if (img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong) \
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong) \
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                            img[i - 1, j + 1] == strong):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError:
                    pass
    return img


# pipeline:

def canny_sketch_pipeline(img_array_rgb, kernel_size=5, sigma=1.4, low_ratio=0.05, high_ratio=0.09):
    # 0. Chuyển đổi rgb sang ảnh xám
    img_gray = rgb2gray(img_array_rgb)

    # 1. Smoothing: làm mịn dùng bộ lọc Gaussian
    kernel = gaussian_kernel(kernel_size, sigma)
    img_gb = ndimage.filters.convolve(img_gray, kernel)

    # 2. Tính Gradient (Sobel)
    img_sobel, theta = sobel_filters(img_gb)

    # 3. Non-Maximum Suppression (Edge Thinning)
    img_nms = edge_thinning(img_sobel, theta)

    # 4. Double Threshold
    img_dt, weak, strong = dbl_threshold(img_nms, low_ratio, high_ratio)

    # 5. Edge Tracking by Hysteresis
    img_final_edges = hysteresis(img_dt, weak, strong)

    # 6. Tạo hiệu ứng vẽ tay (Sketch Effect: Nét đen trên nền trắng)
    img_final_edges_uint8 = img_final_edges.astype(np.uint8)  # -> chuyển sang kiểu nguyên 8 bit (0-255)

    # Đảo ngược màu từ:
    # biên trắng, nền đen -> biên đen, nền trắng
    sketch_result = cv2.bitwise_not(img_final_edges_uint8)

    return sketch_result
