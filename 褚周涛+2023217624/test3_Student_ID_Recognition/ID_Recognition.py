import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def load_model(model_path='mnist_cnn_model.h5'):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 '{model_path}'")
        print("请先运行 train.py 训练模型")
        return None

    model = keras.models.load_model(model_path)
    print(f"成功加载模型: {model_path}")
    return model

def preprocess_image(image_path='image.jpg'):
    """
    预处理学号照片
    """
    if not os.path.exists(image_path):
        print(f"错误: 未找到图片文件 '{image_path}'")
        return None, None

    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 '{image_path}'")
        return None, None

    print(f"原始图像尺寸: {img.shape[1]}x{img.shape[0]}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应二值化
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    return binary, img

def segment_digits(binary_image, min_area=200, expand_ratio=0.3):
    """
    从二值图像中分割出各个数字

    扩展ROI区域，为数字添加边缘填充
    expand_ratio: 扩展比例（0.3表示每边扩展30%）
    """
    height, width = binary_image.shape

    # 寻找轮廓
    contours, _ = cv2.findContours(
        binary_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"检测到原始轮廓: {len(contours)} 个")

    digit_regions = []
    for i, contour in enumerate(contours):
        # 获取最小边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤太小的区域（噪声）
        area = w * h
        if area < min_area:
            continue

        # 计算扩展量（每边扩展 expand_ratio * min(w,h)）
        expand_size = int(min(w, h) * expand_ratio)

        # 计算扩展后的坐标（确保不超出图像边界）
        x_new = max(0, x - expand_size)
        y_new = max(0, y - expand_size)
        x2_new = min(width, x + w + expand_size)
        y2_new = min(height, y + h + expand_size)

        # 使用扩展后的坐标
        w_new = x2_new - x_new
        h_new = y2_new - y_new

        # 提取扩展后的数字区域
        digit_roi = binary_image[y_new:y2_new, x_new:x2_new]

        # 保存区域信息（使用扩展后的坐标）
        digit_regions.append({
            'roi': digit_roi,
            'x': x_new,      # 扩展后的坐标
            'y': y_new,
            'w': w_new,
            'h': h_new,
            'original_w': w, # 保留原始尺寸供参考
            'original_h': h
        })

    # 按x坐标排序（从左到右）
    digit_regions.sort(key=lambda d: d['x'])

    print(f"有效数字区域: {len(digit_regions)} 个")

    # 打印详细信息
    for i, region in enumerate(digit_regions):
        print(f"  数字 {i+1}: 位置({region['x']},{region['y']}) 尺寸{region['w']}x{region['h']} "
              f"(原始尺寸: {region['original_w']}x{region['original_h']})")

    if len(digit_regions) == 0:
        print("警告: 未检测到任何数字区域")

    return digit_regions

def prepare_digit_for_model(digit_roi, target_size=(28, 28)):
    """
    将分割出的数字调整为模型输入格式
    """
    h, w = digit_roi.shape

    # 创建正方形背景
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)

    # 将ROI居中放置在正方形中
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi

    # 调整大小到28x28
    resized = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)

    # 归一化到[0,1]
    normalized = resized.astype('float32') / 255.0

    # 扩展维度以匹配模型输入 (28,28) -> (1,28,28,1)
    input_data = np.expand_dims(np.expand_dims(normalized, 0), -1)

    return input_data

def predict_digits(model, digit_regions):
    """
    预测所有分割出的数字
    """
    if not digit_regions:
        print("错误: 没有可识别的数字区域")
        return []

    predictions = []
    confidences = []

    print("\n开始识别数字...")
    for i, region in enumerate(digit_regions):
        input_data = prepare_digit_for_model(region['roi'])
        pred_probs = model.predict(input_data, verbose=0)

        pred_digit = np.argmax(pred_probs)
        confidence = np.max(pred_probs)

        predictions.append(pred_digit)
        confidences.append(confidence)

        print(f"数字 {i+1:2d}: {pred_digit} (置信度: {confidence:.2%})")

    avg_confidence = np.mean(confidences) if confidences else 0
    print(f"\n平均置信度: {avg_confidence:.2%}")

    return predictions

def visualize_results(original_image, digit_regions, predictions, output_path='result.jpg'):
    """可视化识别结果并保存"""
    result_img = original_image.copy()

    # 添加标题
    student_id = ''.join(map(str, predictions))
    title = f"Result: {student_id}"
    cv2.putText(result_img, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    for i, (region, pred) in enumerate(zip(digit_regions, predictions)):
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # 绘制边界框（使用扩展后的坐标）
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 绘制预测结果
        text = f"{i+1}:{pred}"
        cv2.putText(result_img, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 可选：绘制原始边界框（红色）对比
        orig_w, orig_h = region['original_w'], region['original_h']
        if orig_w and orig_h:
            cv2.rectangle(result_img, (x, y), (x+orig_w, y+orig_h), (0, 0, 255), 1)

    cv2.imwrite(output_path, result_img)
    print(f"\n可视化结果已保存至: {os.path.abspath(output_path)}")
    print("绿色框：扩展后的ROI | 红色框：原始ROI（如有）")

    return result_img

def main():
    image_path = 'image.jpg'
    model_path = 'mnist_cnn_model.h5'

    print(f"识别任务: {os.path.abspath(image_path)}")
    print("=" * 60)

    # 1. 加载模型
    model = load_model(model_path)
    if model is None:
        return

    # 2. 预处理图像
    result = preprocess_image(image_path)
    if result[0] is None:
        return
    binary_img, original_img = result

    # 3. 分割数字（关键改进：扩大ROI）
    # expand_ratio=0.3 表示每边扩展30%的边距
    # 如果数字仍占满ROI，可增大到 0.5 或 0.6
    digit_regions = segment_digits(binary_img, min_area=200, expand_ratio=0.3)

    if len(digit_regions) == 0:
        print("\n" + "="*60)
        print("错误: 未能分割出任何数字！")
        print("="*60)
        return
    elif len(digit_regions) < 10:
        print(f"\n 警告: 仅检测到 {len(digit_regions)} 个数字")

    # 4. 预测数字
    predictions = predict_digits(model, digit_regions)
    if not predictions:
        return

    # 5. 合并学号
    student_id = ''.join(map(str, predictions))
    print("\n" + "=" * 60)
    print(f"识别结果: {student_id}")
    print(f"学号位数: {len(predictions)}")
    print("=" * 60)

    # 6. 可视化
    visualize_results(original_img, digit_regions, predictions)

    # 7. 保存结果
    with open('recognition_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"学号识别结果\n")
        f.write(f"============\n\n")
        f.write(f"图片文件: {os.path.abspath(image_path)}\n")
        f.write(f"识别学号: {student_id}\n")
        f.write(f"学号位数: {len(predictions)}\n")
        f.write(f"模型文件: {os.path.abspath(model_path)}\n")
    print(f"识别结果已保存至: {os.path.abspath('recognition_result.txt')}")

if __name__ == "__main__":
    main()