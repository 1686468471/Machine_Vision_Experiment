import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# 设置随机种子确保可复现
np.random.seed(42)
tf.random.set_seed(42)
# 指定中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


def load_and_preprocess_data() :
    """加载并预处理MNIST数据集"""
    # 定义数据集本地路径
    dataset_path = 'mnist.npz'

    # 检查数据集是否已存在
    if os.path.exists(dataset_path) :
        print(f" 找到本地MNIST数据集: {dataset_path}")
        print(f" 文件大小: {os.path.getsize(dataset_path) / (1024 * 1024):.2f} MB")
    else :
        print(" 未找到本地数据集，自动下载...")
        # 确保目录存在
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    print("正在加载MNIST数据集...")

    # 加载数据（如果本地不存在会自动下载）
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path=dataset_path)

    # 验证数据加载成功
    print(f" 数据集加载成功！")
    print(f"   训练集: {x_train.shape[0]} 个样本")
    print(f"   测试集: {x_test.shape[0]} 个样本")

    # 数据归一化到[0,1]范围
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 增加通道维度 (28,28) -> (28,28,1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def build_cnn_model() :
    """构建卷积神经网络模型"""
    model = keras.Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # 全连接层
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10个数字类别
    ])

    return model


def train_model(model, x_train, y_train):
    """编译并训练模型"""
    # 编译模型
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n模型架构:")
    model.summary()

    # 训练配置
    batch_size = 128
    epochs = 10

    print(f"\n开始训练（批次大小: {batch_size}, 轮数: {epochs}）...")

    # 训练模型
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1,
        shuffle=True
    )

    return history

def evaluate_model(model, x_test, y_test) :
    """评估模型性能"""
    print("\n正在评估测试集性能...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    return test_acc


def plot_training_history(history) :
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 准确率曲线
    ax1.plot(history.history['accuracy'], label='训练准确率')
    ax1.plot(history.history['val_accuracy'], label='验证准确率')
    ax1.set_title('模型准确率')
    ax1.set_xlabel('轮数')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True)

    # 损失曲线
    ax2.plot(history.history['loss'], label='训练损失')
    ax2.plot(history.history['val_loss'], label='验证损失')
    ax2.set_title('模型损失')
    ax2.set_xlabel('轮数')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存至 'training_history.png'")
    plt.close()


def main() :
    # 检查GPU可用性
    if tf.config.list_physical_devices('GPU') :
        print(f"GPU已启用: {tf.config.list_physical_devices('GPU')}")
    else :
        print("使用CPU训练")

    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # 构建模型
    model = build_cnn_model()

    # 训练模型
    history = train_model(model, x_train, y_train)

    # 评估模型
    test_acc = evaluate_model(model, x_test, y_test)

    # 绘制训练曲线
    plot_training_history(history)

    # 保存模型
    model_save_path = 'mnist_cnn_model.h5'
    model.save(model_save_path)
    print(f"\n模型已保存至: {model_save_path}")

    # 保存训练报告
    with open('training_report.txt', 'w', encoding='utf-8') as f :
        f.write(f"MNIST CNN模型训练报告\n")
        f.write(f"====================\n\n")
        f.write(f"测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)\n")
        f.write(f"训练轮数: 10\n")
        f.write(f"批次大小: 128\n")
        f.write(f"优化器: Adam\n")
        f.write(f"模型文件: {model_save_path}\n")
        f.write(f"训练曲线: training_history.png\n")
    print("训练报告已保存至 'training_report.txt'")


if __name__ == "__main__" :
    main()