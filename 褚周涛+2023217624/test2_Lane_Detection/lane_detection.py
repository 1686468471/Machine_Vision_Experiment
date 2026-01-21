import cv2
import numpy as np
import json
import os


class LaneDetector:
    # 车道线检测器类
    def __init__(self, config_path="lane_config.json"):
        """
        初始化检测器，加载配置参数
        config_path: 配置文件路径
        """
        self.config_path = config_path

        # 默认参数
        self.params = {
            "gaussian_kernel": 5,  # 高斯模糊核大小
            "canny_low": 50,  # Canny低阈值
            "canny_high": 150,  # Canny高阈值
            "hough_rho": 1,  # 霍夫距离分辨率
            "hough_theta": np.pi / 180,  # 霍夫角度分辨率
            "hough_threshold": 15,  # 霍夫最少投票数
            "min_line_length": 40,  # 最短线段长度
            "max_line_gap": 20,  # 最大线段间隙
            "roi_top": 0.45,  # ROI顶部比例
            "roi_bottom": 0.95,  # ROI底部比例
        }

        # 加载已保存的配置（如果存在）
        self.load_config()

    def save_config(self) :
        """保存当前参数到配置文件"""
        try :
            config = self.params.copy()
            config["hough_theta"] = float(config["hough_theta"])

            with open(self.config_path, 'w') as f :
                json.dump(config, f, indent=4)
            print(f" 配置已保存到 {self.config_path}")
        except Exception as e :
            print(f" 保存配置失败: {e}")

    def load_config(self) :
        """从配置文件加载参数"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                self.params.update(loaded)
                print(f"配置已从 {self.config_path} 加载")
            except Exception as e :
                print(f"加载配置失败，使用默认值: {e}")

    def create_roi_mask(self, image_shape, return_polygon=False):
        """
        创建感兴趣区域(ROI)掩码
        """
        height, width = image_shape[:2]

        top_y = int(height * self.params["roi_top"])
        bottom_y = int(height * self.params["roi_bottom"])

        roi_polygon = np.array([[
            (int(width * 0.05), bottom_y),  # 左下
            (int(width * 0.95), bottom_y),  # 右下
            (int(width * 0.90), top_y),  # 右上
            (int(width * 0.10), top_y)  # 左上
        ]], dtype=np.int32)

        if return_polygon :
            return roi_polygon

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, roi_polygon, 255)
        return mask

    def detect_edges(self, image) :
        """
        边缘检测：灰度化 → 高斯模糊 → Canny边缘检测

        返回: 二值边缘图像
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = self.params["gaussian_kernel"]
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        edges = cv2.Canny(blurred, self.params["canny_low"], self.params["canny_high"])
        return edges

    def hough_transform(self, edge_image) :
        """
        霍夫变换：从边缘图中检测直线段

        返回: 线段列表 [[x1, y1, x2, y2], ...]
        """
        lines = cv2.HoughLinesP(
            edge_image,
            rho=self.params["hough_rho"],
            theta=self.params["hough_theta"],
            threshold=self.params["hough_threshold"],
            minLineLength=self.params["min_line_length"],
            maxLineGap=self.params["max_line_gap"]
        )
        return lines if lines is not None else []

    def filter_lane_lines(self, lines, image_width):
        """
        过滤和分类车道线
        根据斜率和位置区分左右车道线
        返回: (左车道线列表, 右车道线列表)
        """
        left_lines = []
        right_lines = []

        for line in lines :
            x1, y1, x2, y2 = line[0]

            # 过滤接近水平的线段
            if abs(y2 - y1) < 20 :
                continue

            # 计算斜率
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

            # 左车道：负斜率，在图像左侧
            if slope < -0.5 and x1 < image_width * 0.6 and x2 < image_width * 0.6 :
                left_lines.append((x1, y1, x2, y2, slope))

            # 右车道：正斜率，在图像右侧
            elif slope > 0.5 and x1 > image_width * 0.4 and x2 > image_width * 0.4 :
                right_lines.append((x1, y1, x2, y2, slope))

        return left_lines, right_lines

    def extrapolate_line(self, lines, image_height) :
        """
        车道线外推：将多个短线段拟合成完整车道线

        使用最小二乘法线性回归
        返回: 完整车道线坐标 (x1, y1, x2, y2)
        """
        if not lines :
            return None

        # 收集所有点
        all_x = []
        all_y = []
        for x1, y1, x2, y2, _ in lines :
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        if len(all_x) < 2 :
            return None

        # 拟合直线 x = m*y + b
        coeffs = np.polyfit(all_y, all_x, 1)
        m, b = coeffs[0], coeffs[1]

        # 外推到图像底部和ROI顶部
        y1 = image_height
        y2 = int(image_height * self.params["roi_top"])

        x1 = int(m * y1 + b)
        x2 = int(m * y2 + b)

        return (x1, y1, x2, y2)

    def detect_lanes(self, image_path, output_path=None, show_intermediate=False) :
        """
        完整的车道线检测流程

        image_path: 输入图像路径
        output_path: 输出图像路径
        show_intermediate: 是否显示中间处理步骤
        返回: 带车道线标记的图像
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None :
            raise ValueError(f"无法读取图像: {image_path}")

        height, width = image.shape[:2]
        print(f"成功读取图像: {width}x{height}")

        # 显示原始图像
        if show_intermediate :
            cv2.imshow("Original image", image)
            cv2.waitKey(0)

        # 边缘检测
        edges = self.detect_edges(image)
        if show_intermediate :
            cv2.imshow("Edge detection", edges)
            cv2.waitKey(0)

        # 应用ROI掩码
        roi_mask = self.create_roi_mask(image.shape)
        roi_edges = cv2.bitwise_and(edges, roi_mask)

        if show_intermediate :
            roi_vis = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
            polygon = self.create_roi_mask(image.shape, return_polygon=True)
            cv2.polylines(roi_vis, polygon, True, (0, 255, 0), 2)
            cv2.imshow("ROI", roi_vis)
            cv2.waitKey(0)

        # 霍夫变换检测直线
        lines = self.hough_transform(roi_edges)
        print(f"→ 检测到 {len(lines)} 条原始线段")

        # 过滤车道线
        left_lines, right_lines = self.filter_lane_lines(lines, width)
        print(f"→ 左车道线: {len(left_lines)} 条")
        print(f"→ 右车道线: {len(right_lines)} 条")

        # 外推完整车道线
        left_lane = self.extrapolate_line(left_lines, height)
        right_lane = self.extrapolate_line(right_lines, height)

        # 绘制结果
        result = image.copy()

        # 绘制半透明ROI区域
        overlay = result.copy()
        polygon = self.create_roi_mask(image.shape, return_polygon=True)
        cv2.fillPoly(overlay, polygon, (0, 255, 0))
        cv2.addWeighted(overlay, 0.1, result, 0.9, 0, result)

        # 绘制车道线（红色，粗线）
        if left_lane :
            cv2.line(result, left_lane[:2], left_lane[2 :], (0, 0, 255), 8)
        if right_lane :
            cv2.line(result, right_lane[:2], right_lane[2 :], (0, 0, 255), 8)

        if show_intermediate :
            cv2.imshow("Final Image", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 保存结果
        if output_path :
            cv2.imwrite(output_path, result)
            print(f"✓ 结果已保存到: {output_path}")

        return result


def main() :
    print("=" * 60)
    print("车道线检测系统")
    print("=" * 60)

    # 设置输入输出路径（当前目录）
    input_path = "input.jpg"
    output_path = "output.jpg"

    # 检查输入文件
    if not os.path.exists(input_path) :
        print(f"错误: 找不到 {input_path}")
        return

    # 创建检测器并保存配置
    detector = LaneDetector()
    detector.save_config()

    try :
        print(f"开始检测: {input_path}")
        print("-" * 60)

        # 执行检测（show_intermediate=True 可查看中间步骤）
        result = detector.detect_lanes(
            image_path=input_path,
            output_path=output_path,
            show_intermediate=True  # 改为True可调试图像处理过程
        )

        print("-" * 60)
        print("检测完成!")
        print(f"请查看生成的文件: {output_path}")
        print("=" * 60)

    except Exception as e :
        print(f"检测失败: {e}")
        print("=" * 60)


if __name__ == "__main__" :
    main()
