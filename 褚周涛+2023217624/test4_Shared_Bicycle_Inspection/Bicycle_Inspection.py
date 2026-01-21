from ultralytics import YOLO
import cv2
import numpy as np
import os

def load_model(model_path='runs/detect/bicycle_detection/weights/best.pt'):
    """加载训练好的YOLO模型"""
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 '{model_path}'")
        print("请确认模型文件路径正确")
        return None

    model = YOLO(model_path)
    print(f"成功加载模型: {model_path}")
    print(f"设备: CPU")
    return model

def preprocess_image(image_path='image.jpg'):
    """
    预处理输入图片
    返回: 原始图片和预处理后的图片
    """
    if not os.path.exists(image_path):
        print(f"错误: 未找到图片文件 '{image_path}'")
        return None, None

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 '{image_path}'")
        return None, None

    print(f"原始图像尺寸: {img.shape[1]}x{img.shape[0]}")
    return img

def detect_bicycles(model, image, conf_threshold=0.25, iou_threshold=0.45):
    """
    检测图片中的共享单车
    参数:
        model: YOLO模型
        image: 输入图片
        conf_threshold: 置信度阈值
        iou_threshold: NMS的IOU阈值

    返回:
        detections: 检测结果列表
        annotated_img: 标注后的图片
    """
    results = model(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
        verbose=False
    )

    # 获取结果
    result = results[0]
    detections = []

    # 提取检测框
    boxes = result.boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            # 获取坐标和置信度
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()

            # 存储检测结果
            detection = {
                'id': i + 1,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class': 'bicycle'
            }
            detections.append(detection)

    # 绘制结果
    annotated_img = result.plot()

    return detections, annotated_img

def save_results(detections, annotated_img, output_dir='detection_results'):
    """保存检测结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存可视化结果
    output_img_path = os.path.join(output_dir, 'detected_bicycles.jpg')
    cv2.imwrite(output_img_path, annotated_img)
    print(f"\n可视化结果已保存至: {os.path.abspath(output_img_path)}")

    # 2. 保存位置信息（txt格式）
    result_txt_path = os.path.join(output_dir, 'bicycle_locations.txt')
    with open(result_txt_path, 'w', encoding='utf-8') as f:
        f.write("共享单车检测位置信息\n")
        f.write("=" * 50 + "\n\n")

        if len(detections) == 0:
            f.write("未检测到共享单车\n")
        else:
            f.write(f"检测到的共享单车数量: {len(detections)}\n\n")
            for det in detections:
                f.write(f"单车 {det['id']}:\n")
                f.write(f"  置信度: {det['confidence']:.3f}\n")
                f.write(f"  位置: 左上角({det['bbox'][0]}, {det['bbox'][1]}), 右下角({det['bbox'][2]}, {det['bbox'][3]})\n")
                f.write(f"  中心点: ({(det['bbox'][0]+det['bbox'][2])//2}, {(det['bbox'][1]+det['bbox'][3])//2})\n")
                f.write(f"  宽度: {det['bbox'][2]-det['bbox'][0]}px, 高度: {det['bbox'][3]-det['bbox'][1]}px\n\n")

    print(f"位置信息已保存至: {os.path.abspath(result_txt_path)}")

def main():
    conf_threshold = 0.25
    iou_threshold = 0.45
    image_name = "image.jpg"

    print("=" * 70)
    print("共享单车检测系统")
    print("=" * 70)

    # 1. 加载模型
    model_path = 'runs/detect/bicycle_detection/weights/best.pt'
    model = load_model(model_path)
    if model is None:
        return

    # 2. 加载图片
    image_path = image_name
    original_image = preprocess_image(image_path)
    if original_image is None:
        return

    # 3. 执行检测
    print("\n执行检测...")
    detections, annotated_image = detect_bicycles(model, original_image,conf_threshold,iou_threshold)

    # 4. 输出结果
    print("\n" + "=" * 70)
    print("检测结果")
    print("=" * 70)
    if len(detections) == 0:
        print(" 未检测到共享单车")
    else:
        print(f"检测到 {len(detections)} 辆共享单车")
        for det in detections:
            print(f"  单车{det['id']}: 置信度={det['confidence']:.2%}, " 
                  f"位置=({det['bbox'][0]},{det['bbox'][1]})-({det['bbox'][2]},{det['bbox'][3]})")

    # 5. 保存结果
    save_results(detections, annotated_image)

    # 6. 显示结果
    try:
        cv2.imshow('检测结果', annotated_image)
        print("\n按任意键退出显示窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n无法显示图像窗口: {e}")
        print("请直接查看生成的图片文件")

    print("\n" + "=" * 70)
    print("检测完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()