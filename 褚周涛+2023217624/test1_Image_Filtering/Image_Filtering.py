import cv2
import numpy as np
import os
from typing import Tuple

# ---- 读取图像
image = cv2.imread("image.jpg")

if image is None:
    print("Error: 图像加载失败，请检查路径是否正确。")
    exit()

# ---- 显示原始图像
handle_image = cv2.resize(image, (1024, 576), interpolation=cv2.INTER_AREA)
cv2.imshow('Original Image', handle_image)

# ---- 不调动函数包实现卷积操作
def manual_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray :
    """
    image: 输入图像
    kernel: 卷积核
    return: 卷积结果
    """
    # 获取图像尺寸
    h, w = image.shape[:2]

    # 获取卷积核尺寸
    k_h, k_w = kernel.shape[:2]

    # 计算填充尺寸
    pad_h: int = k_h // 2
    pad_w: int = k_w // 2

    # 创建输出图像
    if len(image.shape) == 3 :  # 彩色图像
        output = np.zeros((h, w, 3), dtype=np.float32)
        # 对每个通道分别进行卷积
        for c in range(3) :
            # 为当前通道添加边界填充
            # 明确指定填充参数的类型
            pad_width = ((pad_h, pad_h), (pad_w, pad_w))
            padded_channel = image[:, :, c].astype(np.float32)
            padded = np.pad(padded_channel, pad_width,
                            mode='constant', constant_values=0)

            # 手动卷积计算
            for i in range(h) :
                for j in range(w) :
                    # 提取当前卷积区域
                    region = padded[i :i + k_h, j :j + k_w]
                    # 卷积计算
                    output[i, j, c] = np.sum(region * kernel)
    else :  # 灰度图像
        output = np.zeros((h, w), dtype=np.float32)
        # 添加边界填充
        pad_width = ((pad_h, pad_h), (pad_w, pad_w))
        padded_image = image.astype(np.float32)
        padded = np.pad(padded_image, pad_width,
                        mode='constant', constant_values=0)

        # 手动卷积计算
        for i in range(h) :
            for j in range(w) :
                # 提取当前卷积区域
                region = padded[i :i + k_h, j :j + k_w]
                # 卷积计算
                output[i, j] = np.sum(region * kernel)

    # 将结果缩放到0-255范围并转换为uint8
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)


# ---- Sobel算子滤波
# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 定义Sobel算子
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)

# 分别进行x方向和y方向的卷积
sobel_x_result = manual_convolution(gray_image, sobel_x)
sobel_y_result = manual_convolution(gray_image, sobel_y)

# 合并x和y方向的梯度
sobel_result = np.sqrt(np.square(sobel_x_result.astype(np.float32)) +
                       np.square(sobel_y_result.astype(np.float32)))
sobel_result = np.clip(sobel_result, 0, 255).astype(np.uint8)

# 显示和保存Sobel滤波结果
sobel_display = cv2.resize(sobel_result, (1024, 576), interpolation=cv2.INTER_AREA)
cv2.imshow('Sobel Filtered Image', sobel_display)

cv2.imwrite('sobel_filtered.jpg', sobel_result)

# ---- 给定卷积核滤波
# 定义给定的卷积核
given_kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype=np.float32)

# 对灰度图像应用给定卷积核
given_filter_result = manual_convolution(gray_image, given_kernel)

# 显示和保存给定卷积核滤波结果
given_filter_display = cv2.resize(given_filter_result, (1024, 576), interpolation=cv2.INTER_AREA)
cv2.imshow('Given Kernel Filtered Image', given_filter_display)

cv2.imwrite('given_kernel_filtered.jpg', given_filter_result)


# ---- 手动计算颜色直方图
def manual_color_histogram(image: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
    """
    image: 输入彩色图像
    bins: 直方图bin数量
    return: 三个通道的直方图
    """
    # 分离BGR三个通道
    b_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    r_channel = image[:, :, 2]

    # 初始化直方图数组
    b_hist = np.zeros(bins, dtype=np.int32)
    g_hist = np.zeros(bins, dtype=np.int32)
    r_hist = np.zeros(bins, dtype=np.int32)

    # 计算每个通道的直方图
    for i in range(bins) :
        b_hist[i] = np.sum(b_channel == i)
        g_hist[i] = np.sum(g_channel == i)
        r_hist[i] = np.sum(r_channel == i)

    return b_hist, g_hist, r_hist


# 计算颜色直方图
b_hist, g_hist, r_hist = manual_color_histogram(image, bins=256)


# 创建可视化直方图图像
def visualize_histogram(b_hist: np.ndarray, g_hist: np.ndarray, r_hist: np.ndarray,
                        img_width: int = 800, img_height: int = 600) -> np.ndarray :
    """
    b_hist: B通道直方图
    g_hist: G通道直方图
    r_hist: R通道直方图
    img_width: 输出图像宽度
    img_height: 输出图像高度
    return: 直方图可视化图像
    """
    # 创建空白图像
    hist_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # 归一化直方图数据以适应图像高度
    max_val = max(np.max(b_hist), np.max(g_hist), np.max(r_hist))
    scale_factor = (img_height - 50) / max_val if max_val > 0 else 1

    # 计算x轴缩放
    bin_count = len(b_hist)
    bin_width = img_width / bin_count

    # 绘制B通道直方图（蓝色）
    for i in range(bin_count - 1) :
        x1 = int(i * bin_width)
        y1 = img_height - int(b_hist[i] * scale_factor)
        x2 = int((i + 1) * bin_width)
        y2 = img_height - int(b_hist[i + 1] * scale_factor)

        # 绘制线
        cv2.line(hist_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 绘制G通道直方图（绿色）
    for i in range(bin_count - 1) :
        x1 = int(i * bin_width)
        y1 = img_height - int(g_hist[i] * scale_factor)
        x2 = int((i + 1) * bin_width)
        y2 = img_height - int(g_hist[i + 1] * scale_factor)

        # 绘制线
        cv2.line(hist_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制R通道直方图（红色）
    for i in range(bin_count - 1) :
        x1 = int(i * bin_width)
        y1 = img_height - int(r_hist[i] * scale_factor)
        x2 = int((i + 1) * bin_width)
        y2 = img_height - int(r_hist[i + 1] * scale_factor)

        # 绘制线
        cv2.line(hist_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 添加坐标轴
    cv2.line(hist_image, (0, img_height - 10), (img_width, img_height - 10), (0, 0, 0), 2)  # x轴
    cv2.line(hist_image, (10, 0), (10, img_height), (0, 0, 0), 2)  # y轴

    # 添加标签
    cv2.putText(hist_image, 'Color Histogram', (img_width // 2 - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(hist_image, 'Pixel Value', (img_width // 2 - 60, img_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(hist_image, 'Frequency', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 添加图例
    cv2.putText(hist_image, 'Blue Channel', (img_width - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(hist_image, 'Green Channel', (img_width - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(hist_image, 'Red Channel', (img_width - 200, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return hist_image


# 生成并显示直方图可视化
hist_visualization = visualize_histogram(b_hist, g_hist, r_hist)
hist_display = cv2.resize(hist_visualization, (1024, 576), interpolation=cv2.INTER_AREA)
cv2.imshow('Color Histogram', hist_display)

cv2.imwrite('color_histogram.jpg', hist_visualization)


# ----手动提取纹理特征（基于LBP算法）
def manual_lbp_texture_features(image: np.ndarray, radius: int = 1, points: int = 8) -> Tuple[np.ndarray, np.ndarray] :
    """
    image: 输入灰度图像
    radius: LBP半径
    points: 邻域点数
    return: LBP特征直方图和LBP图像
    """
    h, w = image.shape
    lbp_image = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)

    # 定义角度
    angles = 2 * np.pi * np.arange(points) / points

    for i in range(radius, h - radius) :
        for j in range(radius, w - radius) :
            # 获取中心像素值
            center = image[i, j]

            # 计算LBP值
            lbp_value = 0
            for p in range(points) :
                # 计算邻域像素坐标
                x = i + int(radius * np.cos(angles[p]))
                y = j - int(radius * np.sin(angles[p]))

                # 获取邻域像素值
                neighbor = image[x, y]

                # 计算二进制位
                if neighbor >= center :
                    lbp_value += 1 << p

            # 存储LBP值
            lbp_image[i - radius, j - radius] = lbp_value

    # 计算LBP直方图（256个bins，因为LBP值范围是0-255）
    hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))

    return hist, lbp_image


# 提取LBP纹理特征
texture_features, lbp_image = manual_lbp_texture_features(gray_image)

# 显示LBP纹理图像
lbp_display = cv2.resize(lbp_image, (1024, 576), interpolation=cv2.INTER_AREA)
cv2.imshow('LBP Texture Image', lbp_display)

cv2.imwrite('lbp_texture.jpg', lbp_image)

# ---- 保存纹理特征到npy格式
# 将纹理特征保存为npy文件
np.save('texture_features.npy', texture_features)
print(f"纹理特征已保存到 texture_features.npy")

# ---- 打印特征信息
print("=" * 50)
print(f"原始图像尺寸: {image.shape}")
print(f"Sobel滤波图像尺寸: {sobel_result.shape}")
print(f"给定卷积核滤波图像尺寸: {given_filter_result.shape}")
print(f"颜色直方图维度: B通道={len(b_hist)}, G通道={len(g_hist)}, R通道={len(r_hist)}")
print("=" * 50)

def test_texture_features(filepath='texture_features.npy') :
    try :
        # 1. 加载npy文件
        features = np.load(filepath)
        print("\n检查纹理特征情况")
        print("=" * 50)
        print(f"成功加载文件: {filepath}")
        print(f"特征形状: {features.shape}")
        print(f"特征数据类型: {features.dtype}")

        # 2. 基本统计信息
        print("\n--- 基本统计信息 ---")
        print(f"特征最小值: {np.min(features)}")
        print(f"特征最大值: {np.max(features)}")
        print(f"特征总和: {np.sum(features)}")
        print(f"特征均值: {np.mean(features):.2f}")
    except FileNotFoundError :
        print(f"错误：文件 {filepath} 未找到")
        return None
    except Exception as e :
        print(f"错误：加载文件时出错 - {e}")
        return None

features = test_texture_features('texture_features.npy')

# ---- 关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
# ---- 显示所有结果摘要
print("\n生成的文件列表:")
files = ['sobel_filtered.jpg', 'given_kernel_filtered.jpg',
         'color_histogram.jpg', 'lbp_texture.jpg', 'texture_features.npy']

# sobel_filtered.jpg
if os.path.exists(files[0]):
    print("sobel_filtered.jpg(使用Sobel算子处理后的图像) 正常存在")
else:
    print("sobel_filtered.jpg(使用Sobel算子处理后的图像) 未找到")

# given_kernel_filtered.jpg
if os.path.exists(files[1]):
    print("given_kernel_filtered.jpg(使用特定卷积核(1 0 -1; 2 0 -2; 1 0 -1)处理后的图像) 正常存在")
else:
    print("given_kernel_filtered.jpg(使用特定卷积核(1 0 -1; 2 0 -2; 1 0 -1)处理后的图像) 未找到")

# color_histogram.jpg
if os.path.exists(files[2]):
    print("color_histogram.jpg(颜色直方图可视化) 正常存在")
else:
    print("color_histogram.jpg(颜色直方图可视化) 未找到")

# lbp_texture.jpg
if os.path.exists(files[3]):
    print("lbp_texture.jpg(局部二值模式(LBP)算法生成的纹理图像) 正常存在")
else:
    print("lbp_texture.jpg(局部二值模式(LBP)算法生成的纹理图像) 未找到")

# texture_features.npy
if os.path.exists(files[4]):
    print("texture_features.npy(纹理特征数据文件) 正常存在")
else:
    print("texture_features.npy(纹理特征数据文件) 未找到")
