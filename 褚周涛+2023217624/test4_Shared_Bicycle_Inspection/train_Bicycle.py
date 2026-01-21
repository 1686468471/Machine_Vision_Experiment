from ultralytics import YOLO
import torch
import os


def check_gpu_environment() :
    """检查GPU环境"""
    print("=" * 60)
    print("GPU环境检测")
    print("=" * 60)

    if torch.cuda.is_available() :
        gpu_name = torch.cuda.get_device_name(0)
        print(f"检测到GPU: {gpu_name}")
        device = 'cuda:0'
        torch.cuda.empty_cache()
    else :
        print(" 未检测到GPU，将使用CPU训练")
        device = 'cpu'

    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {device}")
    print("=" * 60)

    return device


def train_model(epochs=15, imgsz=640, batch=16) :
    """训练模型（核心函数）"""

    # 1. 检查GPU
    device = check_gpu_environment()

    # 2. 验证数据路径
    coco_root = 'coco_dataset'
    data_yaml = os.path.join(coco_root, 'coco_bicycle_all.yaml')

    if not os.path.exists(data_yaml) :
        print(f" 找不到配置文件: {data_yaml}")
        print("请确保已完成COCO到YOLO的转换")
        return None

    # 验证标签目录
    train_label_dir = os.path.join(coco_root, 'labels', 'train2017')
    if not os.path.exists(train_label_dir) :
        print(f" 找不到标签目录: {train_label_dir}")
        print("请运行: python convert_coco_to_yolo.py")
        return None

    # 统计标签文件数量
    label_files = [f for f in os.listdir(train_label_dir) if f.endswith('.txt')]
    print(f"\n 找到 {len(label_files)} 个训练标签文件")

    if len(label_files) == 0 :
        print(" 标签目录为空！")
        return None

    # 3. 打印训练配置
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"数据集: {data_yaml}")
    print(f"训练轮数: {epochs}")
    print(f"图片尺寸: {imgsz}")
    print(f"批次大小: {batch}")
    print(f"设备: {device}")
    print("=" * 60)

    # 4. 加载模型
    print("\n加载YOLO模型...")
    model = YOLO('yolov8n.pt')

    # 5. 开始训练
    try :
        print("\n开始训练...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='bicycle_detection',
            save=True,
            save_period=5,  # 每5轮保存一次
            device=device,
            workers=8,
            amp=True,
            cache=False,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最佳模型: runs/detect/bicycle_detection/weights/best.pt")
        print("=" * 60)

        return model

    except RuntimeError as e :
        if "CUDA out of memory" in str(e) :
            print("\n 显存不足！尝试减小batch或imgsz")
        else :
            print(f"\n 训练失败: {e}")
        return None


def main() :
    print("共享单车目标检测训练系统")
    print("=" * 60)
    print("模式: 直接使用已转换的YOLO标注")
    print("=" * 60)

    # 直接开始训练
    model = train_model(epochs=15, imgsz=640, batch=64)

    if model is None :
        print("训练失败")
        return

    # 导出模型
    print("\n导出ONNX模型...")
    try :
        model.export(format='onnx')
        print("导出成功")
    except Exception as e :
        print(f"导出失败: {e}")

    print("\n" + "=" * 60)
    print("所有步骤完成！")
    print("=" * 60)


if __name__ == "__main__" :
    main()