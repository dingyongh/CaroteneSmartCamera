import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
import os
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文显示，仅使用SimHei
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class HSVChangeAnalyzer:
    def __init__(self, parent_frame, on_train_complete):
        self.parent_frame = parent_frame
        self.on_train_complete = on_train_complete  # 训练完成回调函数

        # 数据存储
        self.image_folder = ""
        self.image_files = []  # 存储图片文件路径和时间
        self.time_offsets = []  # 相对于基准时间的偏移量(秒)
        self.hsv_data = []  # 存储每张图片的H(色调), S(饱和度), V(亮度)值
        self.models = {}  # 存储H, S, V三个通道的模型
        self.polynomial_degree = 2  # 默认多项式次数
        self.reference_time = None  # 基准时间

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 清空父框架
        for widget in self.parent_frame.winfo_children():
            widget.destroy()

        # 左侧控制面板
        left_frame = ttk.Frame(self.parent_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 右侧图表显示区
        right_frame = ttk.Frame(self.parent_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧内容 - 文件夹选择
        folder_frame = ttk.Frame(left_frame)
        folder_frame.pack(fill=tk.X, pady=5)

        ttk.Label(folder_frame, text="图片目录:").pack(side=tk.LEFT, padx=5)
        self.folder_path_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_path_var, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(folder_frame, text="浏览", command=self.browse_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_frame, text="加载图片", command=self.load_images).pack(side=tk.LEFT, padx=5)

        # 左侧内容 - 多项式次数选择
        degree_frame = ttk.Frame(left_frame)
        degree_frame.pack(fill=tk.X, pady=5)

        ttk.Label(degree_frame, text="多项式回归次数 (2-5):").pack(side=tk.LEFT, padx=5)
        self.degree_var = tk.IntVar(value=self.polynomial_degree)
        degree_combo = ttk.Combobox(degree_frame, textvariable=self.degree_var, values=[2,3,4,5], width=5)
        degree_combo.pack(side=tk.LEFT, padx=5)
        degree_combo.bind("<<ComboboxSelected>>", self.update_degree)

        # 左侧内容 - 图片列表
        list_frame = ttk.LabelFrame(left_frame, text="图片列表")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)

        # 左侧内容 - 训练按钮
        ttk.Button(left_frame, text="训练模型", command=self.train_model).pack(fill=tk.X, pady=5)

        # 右侧内容 - HSV变化曲线
        self.curve_frame = ttk.LabelFrame(right_frame, text="HSV变化曲线 (H:色调, S:饱和度, V:亮度)")
        self.curve_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.curve_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path_var.set(folder)
            self.image_folder = folder

    def load_images(self):
        if not self.folder_path_var.get():
            messagebox.showerror("错误", "请先选择图片目录")
            return

        self.image_folder = self.folder_path_var.get()
        self.image_files = []
        self.image_listbox.delete(0, tk.END)

        # 正则表达式匹配文件名格式
        pattern = r'^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.jpg$'

        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith('.jpg'):
                match = re.match(pattern, filename)
                if match:
                    try:
                        # 解析日期时间
                        year, month, day, hour, minute, second = map(int, match.groups())
                        dt = datetime(year, month, day, hour, minute, second)
                        self.image_files.append((os.path.join(self.image_folder, filename), dt, filename))
                    except ValueError:
                        print(f"忽略无效的日期格式文件: {filename}")

        # 按时间排序
        self.image_files.sort(key=lambda x: x[1])

        if not self.image_files:
            messagebox.showinfo("信息", "未找到符合格式的JPG图片")
            return

        # 显示文件列表
        for _, _, filename in self.image_files:
            self.image_listbox.insert(tk.END, filename)

        # 设置基准时间
        self.reference_time = self.image_files[0][1]
        messagebox.showinfo("信息", f"成功加载 {len(self.image_files)} 张图片，基准时间: {self.reference_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 提取时间偏移和HSV数据
        self.time_offsets = []
        self.hsv_data = []

        for img_path, dt, _ in self.image_files:
            # 计算时间偏移(秒)
            offset = (dt - self.reference_time).total_seconds()
            self.time_offsets.append(offset)

            # 读取图片并转换为HSV颜色空间
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)

                # 将RGB转换为HSV
                hsv_img = Image.fromarray(img_array).convert('HSV')
                hsv_array = np.array(hsv_img)

                # 计算H, S, V三个通道的平均值
                avg_h = np.mean(hsv_array[:, :, 0])  # 色调 (0-255)
                avg_s = np.mean(hsv_array[:, :, 1])  # 饱和度 (0-255)
                avg_v = np.mean(hsv_array[:, :, 2])  # 亮度 (0-255)

                self.hsv_data.append((avg_h, avg_s, avg_v))
            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {e}")
                # 移除对应的时间偏移
                self.time_offsets.pop()

    def update_degree(self, event=None):
        selected_degree = self.degree_var.get()
        if 2 <= selected_degree <= 5:
            self.polynomial_degree = selected_degree
        else:
            messagebox.showerror("错误", "请选择2-5之间的多项式次数")
            self.degree_var.set(self.polynomial_degree)

    def train_model(self):
        if not self.time_offsets or not self.hsv_data:
            messagebox.showerror("错误", "请先加载图片")
            self.on_train_complete(False)  # 训练失败
            return

        try:
            # 准备数据
            X = np.array(self.time_offsets).reshape(-1, 1)
            h_values = np.array([hsv[0] for hsv in self.hsv_data])  # 色调
            s_values = np.array([hsv[1] for hsv in self.hsv_data])  # 饱和度
            v_values = np.array([hsv[2] for hsv in self.hsv_data])  # 亮度

            # 为每个通道创建多项式回归模型
            self.models['h'] = make_pipeline(
                PolynomialFeatures(degree=self.polynomial_degree, include_bias=False),
                LinearRegression()
            )
            self.models['s'] = make_pipeline(
                PolynomialFeatures(degree=self.polynomial_degree, include_bias=False),
                LinearRegression()
            )
            self.models['v'] = make_pipeline(
                PolynomialFeatures(degree=self.polynomial_degree, include_bias=False),
                LinearRegression()
            )

            # 训练模型
            self.models['h'].fit(X, h_values)
            self.models['s'].fit(X, s_values)
            self.models['v'].fit(X, v_values)

            # 计算R²分数
            h_r2 = self.models['h'].score(X, h_values)
            s_r2 = self.models['s'].score(X, s_values)
            v_r2 = self.models['v'].score(X, v_values)

            # 绘制HSV变化曲线和回归线
            self.ax.clear()

            # 绘制原始数据点
            self.ax.scatter(self.time_offsets, h_values, color='red', alpha=0.4, s=15, label=f'色调 (R²={h_r2:.4f})')
            self.ax.scatter(self.time_offsets, s_values, color='green', alpha=0.4, s=15, label=f'饱和度 (R²={s_r2:.4f})')
            self.ax.scatter(self.time_offsets, v_values, color='blue', alpha=0.4, s=15, label=f'亮度 (R²={v_r2:.4f})')

            # 绘制回归线
            X_range = np.linspace(min(self.time_offsets), max(self.time_offsets), 100).reshape(-1, 1)

            h_pred = self.models['h'].predict(X_range)
            s_pred = self.models['s'].predict(X_range)
            v_pred = self.models['v'].predict(X_range)

            self.ax.plot(X_range, h_pred, 'r-', linewidth=2)
            self.ax.plot(X_range, s_pred, 'g-', linewidth=2)
            self.ax.plot(X_range, v_pred, 'b-', linewidth=2)

            self.ax.set_xlabel('时间偏移 (秒)')
            self.ax.set_ylabel('HSV值 (0-255)')
            self.ax.set_title(f'HSV随时间变化曲线 (多项式次数: {self.polynomial_degree})')
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.fig.tight_layout()
            self.canvas.draw()

            messagebox.showinfo("信息",
                              f"模型训练完成\n多项式次数: {self.polynomial_degree}\n"
                              f"色调R²分数: {h_r2:.4f}\n"
                              f"饱和度R²分数: {s_r2:.4f}\n"
                              f"亮度R²分数: {v_r2:.4f}")

            self.on_train_complete(True)  # 训练成功
        except Exception as e:
            messagebox.showerror("错误", f"模型训练失败: {str(e)}")
            self.on_train_complete(False)  # 训练失败


class ProductionModelAnalyzer:
    def __init__(self, parent_frame, on_train_complete):
        self.parent_frame = parent_frame
        self.on_train_complete = on_train_complete

        # 数据和模型变量初始化
        self.df = None
        self.model_production = None
        self.model_index = None
        self.poly_features = None

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 清空父框架
        for widget in self.parent_frame.winfo_children():
            widget.destroy()

        # 创建左中右三个框架，各占三分之一宽度
        total_width = self.parent_frame.winfo_width() if self.parent_frame.winfo_width() > 0 else 1200

        # 左侧框架 - 占1/3宽度
        left_frame = ttk.Frame(self.parent_frame, width=total_width//3, padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=2, pady=2)
        left_frame.pack_propagate(False)  # 保持固定大小

        # 中间框架 - 占1/3宽度
        middle_frame = ttk.Frame(self.parent_frame, width=total_width//3, padding="5")
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=2, pady=2)
        middle_frame.pack_propagate(False)  # 保持固定大小

        # 右侧框架 - 占1/3宽度
        right_frame = ttk.Frame(self.parent_frame, width=total_width//3, padding="5")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=2, pady=2)
        right_frame.pack_propagate(False)  # 保持固定大小

        # ---------------------- 左侧区域组件 ----------------------
        # 顶部按钮区域
        btn_frame = ttk.Frame(left_frame, padding="5")
        btn_frame.pack(fill=tk.X)

        # 加载文件按钮
        self.load_btn = ttk.Button(btn_frame, text="加载Excel文件", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # 多项式阶数选择
        ttk.Label(btn_frame, text="多项式阶数:").pack(side=tk.LEFT, padx=5)
        self.degree_var = tk.IntVar(value=2)
        degree_combo = ttk.Combobox(btn_frame, textvariable=self.degree_var, values=[1,2,3,4,5], width=5)
        degree_combo.pack(side=tk.LEFT, padx=5)

        # 训练模型按钮（放在左侧下方）
        self.train_btn = ttk.Button(left_frame, text="训练模型", command=self.train_model, state=tk.DISABLED)
        self.train_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=5)

        # 中间数据显示区域（填充剩余空间）
        data_frame = ttk.LabelFrame(left_frame, text="数据预览", padding="5")
        data_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        # 创建表格显示数据
        self.tree = ttk.Treeview(data_frame)
        scroll_y = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scroll_x = ttk.Scrollbar(data_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # ---------------------- 中间区域组件 ----------------------
        # 上方曲线图 - 产量模型
        top_graph_frame = ttk.LabelFrame(middle_frame, text="产量模型拟合曲线", padding="5")
        top_graph_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        # 下方曲线图 - 指标数据模型
        bottom_graph_frame = ttk.LabelFrame(middle_frame, text="指标数据模型拟合曲线", padding="5")
        bottom_graph_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        # 创建图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 8), dpi=100)

        # 上方画布
        self.canvas_top = FigureCanvasTkAgg(self.fig, master=top_graph_frame)
        self.canvas_top.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 下方画布（共享同一图形）
        self.canvas_bottom = FigureCanvasTkAgg(self.fig, master=bottom_graph_frame)
        self.canvas_bottom.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---------------------- 右侧区域组件 ----------------------
        # 上方 - 产量模型参数
        top_model_frame = ttk.LabelFrame(right_frame, text="产量模型参数", padding="5")
        top_model_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        # 下方 - 指标数据模型参数
        bottom_model_frame = ttk.LabelFrame(right_frame, text="指标数据模型参数", padding="5")
        bottom_model_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        # 产量模型参数表格
        self.model_tree1 = ttk.Treeview(top_model_frame, columns=["param", "value"], show="headings")
        self.model_tree1.heading("param", text="参数")
        self.model_tree1.heading("value", text="值")
        self.model_tree1.column("param", width=100)
        self.model_tree1.column("value", width=100)

        # 添加滚动条
        scroll1 = ttk.Scrollbar(top_model_frame, orient=tk.VERTICAL, command=self.model_tree1.yview)
        self.model_tree1.configure(yscrollcommand=scroll1.set)
        scroll1.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_tree1.pack(fill=tk.BOTH, expand=True)

        # 指标数据模型参数表格
        self.model_tree2 = ttk.Treeview(bottom_model_frame, columns=["param", "value"], show="headings")
        self.model_tree2.heading("param", text="参数")
        self.model_tree2.heading("value", text="值")
        self.model_tree2.column("param", width=100)
        self.model_tree2.column("value", width=100)

        # 添加滚动条
        scroll2 = ttk.Scrollbar(bottom_model_frame, orient=tk.VERTICAL, command=self.model_tree2.yview)
        self.model_tree2.configure(yscrollcommand=scroll2.set)
        scroll2.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_tree2.pack(fill=tk.BOTH, expand=True)

    def load_file(self):
        """加载Excel文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx;*.xls")]
        )

        if not file_path:
            return

        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            self.df = excel_file.parse(sheet_names[0])  # 取第一个工作表

            # 检查是否有必要的列
            required_columns = ['运行时间', '产量', '指标数据']
            for col in required_columns:
                if col not in self.df.columns:
                    messagebox.showerror("错误", f"Excel文件中缺少必要的列: {col}")
                    self.df = None
                    return

            # 更新表格显示
            self._update_data_table()

            # 启用训练按钮
            self.train_btn.config(state=tk.NORMAL)

            messagebox.showinfo("成功", "文件加载成功")

        except Exception as e:
            messagebox.showerror("错误", f"加载文件时出错: {str(e)}")

    def _update_data_table(self):
        """更新数据表格显示"""
        # 清空现有内容
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 设置列
        self.tree["columns"] = list(self.df.columns)
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        # 添加数据行（最多显示50行）
        display_rows = min(50, len(self.df))
        for i in range(display_rows):
            row = self.df.iloc[i].tolist()
            # 转换为字符串，处理NaN
            row_str = [str(x) if pd.notna(x) else "NaN" for x in row]
            self.tree.insert("", tk.END, values=row_str)

    def train_model(self):
        """训练多项式回归模型"""
        if self.df is None:
            self.on_train_complete(False)
            return

        try:
            # 获取多项式阶数
            degree = self.degree_var.get()

            # 定义自变量和因变量
            X = self.df[['运行时间']]
            y_production = self.df['产量']
            y_index = self.df['指标数据']

            # 创建多项式特征
            self.poly_features = PolynomialFeatures(degree=degree)
            X_poly = self.poly_features.fit_transform(X)

            # 建立运行时间与产量的多项式回归模型
            self.model_production = LinearRegression()
            self.model_production.fit(X_poly, y_production)

            # 建立运行时间与指标数据的多项式回归模型
            self.model_index = LinearRegression()
            self.model_index.fit(X_poly, y_index)

            # 进行预测用于绘图
            y_production_pred = self.model_production.predict(X_poly)
            y_index_pred = self.model_index.predict(X_poly)

            # 评估模型性能
            mse_production = mean_squared_error(y_production, y_production_pred)
            r2_production = r2_score(y_production, y_production_pred)
            mse_index = mean_squared_error(y_index, y_index_pred)
            r2_index = r2_score(y_index, y_index_pred)

            # 更新图形
            self._update_graphs(X, y_production, y_production_pred, y_index, y_index_pred)

            # 更新模型参数显示
            self._update_model_params(mse_production, r2_production, mse_index, r2_index, degree)

            messagebox.showinfo("成功", "模型训练完成")
            self.on_train_complete(True)  # 训练成功

        except Exception as e:
            messagebox.showerror("错误", f"训练模型时出错: {str(e)}")
            self.on_train_complete(False)  # 训练失败

    def _update_graphs(self, X, y_prod, y_prod_pred, y_idx, y_idx_pred):
        """更新曲线图"""
        # 清空现有图形
        self.ax1.clear()
        self.ax2.clear()

        # 绘制运行时间与产量的散点图和回归线
        self.ax1.scatter(X, y_prod, alpha=0.5, label='实际数据')
        sorted_indices = X.values.ravel().argsort()
        self.ax1.plot(X.values.ravel()[sorted_indices], y_prod_pred[sorted_indices], 'r-', label='回归线')
        self.ax1.set_title('运行时间与产量的关系')
        self.ax1.set_xlabel('运行时间')
        self.ax1.set_ylabel('产量')
        self.ax1.legend()

        # 绘制运行时间与指标数据的散点图和回归线
        self.ax2.scatter(X, y_idx, alpha=0.5, label='实际数据')
        self.ax2.plot(X.values.ravel()[sorted_indices], y_idx_pred[sorted_indices], 'r-', label='回归线')
        self.ax2.set_title('运行时间与指标数据的关系')
        self.ax2.set_xlabel('运行时间')
        self.ax2.set_ylabel('指标数据')
        self.ax2.legend()

        # 调整布局并更新画布
        self.fig.tight_layout()
        self.canvas_top.draw()
        self.canvas_bottom.draw()

    def _update_model_params(self, mse_prod, r2_prod, mse_idx, r2_idx, degree):
        """更新模型参数显示"""
        # 清空现有内容
        for item in self.model_tree1.get_children():
            self.model_tree1.delete(item)
        for item in self.model_tree2.get_children():
            self.model_tree2.delete(item)

        # 显示产量模型参数
        self.model_tree1.insert("", tk.END, values=["多项式阶数", degree])
        self.model_tree1.insert("", tk.END, values=["均方误差 (MSE)", f"{mse_prod:.4f}"])
        self.model_tree1.insert("", tk.END, values=["决定系数 (R²)", f"{r2_prod:.4f}"])
        self.model_tree1.insert("", tk.END, values=["截距", f"{self.model_production.intercept_:.4f}"])

        for i, coef in enumerate(self.model_production.coef_):
            self.model_tree1.insert("", tk.END, values=[f"系数 x^{i}", f"{coef:.4f}"])

        # 显示指标数据模型参数
        self.model_tree2.insert("", tk.END, values=["多项式阶数", degree])
        self.model_tree2.insert("", tk.END, values=["均方误差 (MSE)", f"{mse_idx:.4f}"])
        self.model_tree2.insert("", tk.END, values=["决定系数 (R²)", f"{r2_idx:.4f}"])
        self.model_tree2.insert("", tk.END, values=["截距", f"{self.model_index.intercept_:.4f}"])

        for i, coef in enumerate(self.model_index.coef_):
            self.model_tree2.insert("", tk.END, values=[f"系数 x^{i}", f"{coef:.4f}"])


class CombinedInferenceAnalyzer:
    def __init__(self, parent_frame, production_analyzer, on_inference_complete, on_prediction_complete):
        self.parent_frame = parent_frame
        self.production_analyzer = production_analyzer  # 产量指标模型训练器
        self.on_inference_complete = on_inference_complete  # 图像推理完成回调
        self.on_prediction_complete = on_prediction_complete  # 产量指标推测完成回调

        self.models = {}  # 存储H, S, V三个通道的模型
        self.polynomial_degree = 2
        self.reference_time = None
        self.time_offsets = []  # 从HSVChangeAnalyzer获取的时间偏移

        # 推理相关变量
        self.current_image = None
        self.current_image_path = ""
        self.roi_points = []
        self.roi_rect = None
        self.inference_status = "未推理"
        self.inferred_time = None  # 存储推理得到的时间
        self.prediction_status = "未推测"  # 产量指标推测状态

        self.create_widgets()

    def create_widgets(self):
        # 清空父框架
        for widget in self.parent_frame.winfo_children():
            widget.destroy()

        # 设置左中右三栏，平均分布且宽度固定
        total_width = 1200  # 固定总宽度
        column_width = total_width // 3

        # 左侧框架 - 载入图片和ROI操作
        left_frame = ttk.LabelFrame(self.parent_frame, text="图像处理区", padding="10", width=column_width)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_frame.pack_propagate(False)

        # 中间框架 - HSV信息和图像推理
        middle_frame = ttk.LabelFrame(self.parent_frame, text="图像推理区", padding="10", width=column_width)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        middle_frame.pack_propagate(False)

        # 右侧框架 - 产量指标推测
        right_frame = ttk.LabelFrame(self.parent_frame, text="产量指标推测", padding="10", width=column_width)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_frame.pack_propagate(False)

        # ---------------------- 左侧内容 - 载入图片和ROI操作 ----------------------
        # 推理图片加载
        img_frame = ttk.Frame(left_frame)
        img_frame.pack(fill=tk.X, pady=5)

        ttk.Label(img_frame, text="推理图片:", font=("SimHei", 10)).pack(side=tk.LEFT, padx=5)
        self.infer_img_path_var = tk.StringVar()
        ttk.Entry(img_frame, textvariable=self.infer_img_path_var, width=20).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(img_frame, text="浏览", command=self.browse_infer_image).pack(side=tk.LEFT, padx=5)

        # 图片显示和ROI设置
        self.display_frame = ttk.LabelFrame(left_frame, text="图片显示 (点击设置ROI区域)", padding="10")
        self.display_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.display_canvas = tk.Canvas(self.display_frame, bg="gray")
        self.display_canvas.pack(fill=tk.BOTH, expand=True)
        self.display_canvas.bind("<Button-1>", self.on_canvas_click)
        self.display_canvas.bind("<Configure>", self.on_canvas_configure)

        # ROI控制按钮
        roi_btn_frame = ttk.Frame(left_frame)
        roi_btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(roi_btn_frame, text="设置ROI区域", command=self.start_roi_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(roi_btn_frame, text="取消ROI", command=self.clear_roi).pack(side=tk.LEFT, padx=5)

        # ---------------------- 中间内容 - HSV信息和图像推理 ----------------------
        # 图像HSV信息
        hsv_frame = ttk.LabelFrame(middle_frame, text="图像HSV信息", padding="10")
        hsv_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.hsv_info_var = tk.StringVar(value="请加载图片并设置ROI区域")
        ttk.Label(hsv_frame, textvariable=self.hsv_info_var, font=("SimHei", 10), wraplength=300).pack(pady=10, anchor=tk.W)

        # 推理按钮
        ttk.Button(middle_frame, text="图像推理", command=self.perform_inference).pack(fill=tk.X, pady=10)

        # 推理结果显示
        result_frame = ttk.LabelFrame(middle_frame, text="图像推理结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.inference_result_var = tk.StringVar(value="请先加载图片、设置ROI区域并确保已训练模型")
        ttk.Label(
            result_frame,
            textvariable=self.inference_result_var,
            font=("SimHei", 10),
            wraplength=300,
            justify=tk.LEFT
        ).pack(pady=10, anchor=tk.NW)

        # ---------------------- 右侧内容 - 产量指标推测 ----------------------
        # 推测按钮
        ttk.Button(right_frame, text="产量指标推测", command=self.predict_production).pack(fill=tk.X, pady=10)

        # 运行时间显示
        time_frame = ttk.LabelFrame(right_frame, text="使用的运行时间", padding="10")
        time_frame.pack(fill=tk.X, pady=5)

        self.time_var = tk.StringVar(value="等待图像推理结果...")
        ttk.Label(time_frame, textvariable=self.time_var, font=("SimHei", 10)).pack(fill=tk.X, padx=5, pady=5)

        # 产量指标推测结果
        prediction_frame = ttk.LabelFrame(right_frame, text="产量指标推测结果", padding="10")
        prediction_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.prediction_var = tk.StringVar(value="请先完成图像推理")
        ttk.Label(
            prediction_frame,
            textvariable=self.prediction_var,
            font=("SimHei", 10),
            wraplength=300,
            justify=tk.LEFT
        ).pack(pady=10, anchor=tk.NW)

        # 初始化状态
        self.roi_selecting = False
        self.tk_image = None
        self.image_x = 0
        self.image_y = 0
        self.image_scale = 1.0

    # 以下是从InferenceAnalyzer类移植的方法，保持功能不变
    def browse_infer_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("JPG图片", "*.jpg")])
        if file_path:
            self.infer_img_path_var.set(file_path)
            self.load_infer_image(file_path)
            self.inference_status = "未推理"

    def load_infer_image(self, file_path):
        self.current_image_path = file_path
        try:
            img = Image.open(file_path).convert('RGB')
            self.current_image = img.copy()
            self.display_image(img)
            self.hsv_info_var.set("请设置ROI区域以获取HSV信息")

        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {e}")
            self.current_image = None
            self.current_image_path = ""

    def on_canvas_configure(self, event):
        if self.current_image:
            self.display_image(self.current_image)

    def display_image(self, img):
        self.display_canvas.delete("all")
        canvas_width = self.display_canvas.winfo_width()
        canvas_height = self.display_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 300
            canvas_height = 200

        img_width, img_height = img.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(resized_img)
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2

        self.image_x = x
        self.image_y = y
        self.image_scale = ratio

        self.display_canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)

        if self.roi_rect:
            self.draw_roi()

    def start_roi_selection(self):
        if not self.current_image:
            messagebox.showerror("错误", "请先加载推理图片")
            return

        self.roi_selecting = True
        self.roi_points = []
        self.roi_rect = None
        self.display_canvas.delete("roi_rect")
        messagebox.showinfo("信息", "请在图片上点击两个点来确定ROI区域（左上角和右下角）")
        self.inference_status = "未推理"

    def on_canvas_click(self, event):
        if not self.roi_selecting or not self.current_image:
            return

        x = int((event.x - self.image_x) / self.image_scale)
        y = int((event.y - self.image_y) / self.image_scale)

        img_width, img_height = self.current_image.size
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))

        self.roi_points.append((x, y))

        canvas_x = int(x * self.image_scale) + self.image_x
        canvas_y = int(y * self.image_scale) + self.image_y
        self.display_canvas.create_oval(
            canvas_x - 5, canvas_y - 5,
            canvas_x + 5, canvas_y + 5,
            fill="red", tags="roi_points"
        )

        if len(self.roi_points) == 2:
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            self.roi_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.roi_selecting = False

            self.draw_roi()
            self.display_canvas.delete("roi_points")
            self.calculate_roi_hsv()

    def draw_roi(self):
        if not self.roi_rect or not hasattr(self, 'tk_image'):
            return

        x1 = int(self.roi_rect[0] * self.image_scale) + self.image_x
        y1 = int(self.roi_rect[1] * self.image_scale) + self.image_y
        x2 = int(self.roi_rect[2] * self.image_scale) + self.image_x
        y2 = int(self.roi_rect[3] * self.image_scale) + self.image_y

        self.display_canvas.delete("roi_rect")
        self.display_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="roi_rect")
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.display_canvas.create_text(cx, cy, text="ROI", fill="red", font=("SimHei", 12, "bold"), tags="roi_rect")

    def clear_roi(self):
        self.roi_points = []
        self.roi_rect = None
        self.roi_selecting = False
        self.display_canvas.delete("roi_rect")
        self.display_canvas.delete("roi_points")
        self.hsv_info_var.set("请设置ROI区域以获取HSV信息")
        self.inference_status = "未推理"
        messagebox.showinfo("信息", "ROI区域已清除")

    def calculate_roi_hsv(self):
        if not self.roi_rect or not self.current_image:
            return

        x1, y1, x2, y2 = self.roi_rect
        roi = self.current_image.crop((x1, y1, x2, y2))

        roi_array = np.array(roi)
        hsv_roi = Image.fromarray(roi_array).convert('HSV')
        hsv_array = np.array(hsv_roi)

        avg_h = np.mean(hsv_array[:, :, 0])
        avg_s = np.mean(hsv_array[:, :, 1])
        avg_v = np.mean(hsv_array[:, :, 2])

        info = f"ROI区域信息:\n"
        info += f"区域大小: {x2-x1}x{y2-y1} 像素\n"
        info += f"平均色调(H): {avg_h:.2f}\n"
        info += f"平均饱和度(S): {avg_s:.2f}\n"
        info += f"平均亮度(V): {avg_v:.2f}"
        self.hsv_info_var.set(info)

        return avg_h, avg_s, avg_v

    def perform_inference(self):
        if not self.models or len(self.models) != 3:
            messagebox.showerror("错误", "请先进行设备运行时间模型训练")
            self.inference_status = "失败"
            self.on_inference_complete(False)
            return

        if not self.current_image:
            messagebox.showerror("错误", "请先加载推理图片")
            self.inference_status = "失败"
            self.on_inference_complete(False)
            return

        if not self.roi_rect:
            messagebox.showerror("错误", "请先设置ROI区域")
            self.inference_status = "失败"
            self.on_inference_complete(False)
            return

        try:
            avg_h, avg_s, avg_v = self.calculate_roi_hsv()

            def predict_time(target_value, model):
                if not self.time_offsets or len(self.time_offsets) == 0:
                    raise ValueError("没有训练数据的时间偏移信息")

                min_time = min(self.time_offsets)
                max_time = max(self.time_offsets)
                extended_min = min(0, min_time * 1.1)
                extended_max = max_time * 1.1

                time_candidates = np.linspace(extended_min, extended_max, 10000).reshape(-1, 1)
                predictions = model.predict(time_candidates)
                closest_idx = np.argmin(np.abs(predictions - target_value))
                predicted_offset = time_candidates[closest_idx][0]
                return max(0, predicted_offset)

            h_offset = predict_time(avg_h, self.models['h'])
            s_offset = predict_time(avg_s, self.models['s'])
            final_offset = (h_offset + s_offset) / 2

            # 保存推理得到的时间
            self.inferred_time = final_offset
            self.update_time_display()

            # 格式化结果
            def format_offset(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours}小时 {minutes}分钟 {secs}秒 ({seconds:.1f}秒)"

            result = f"推理结果:\n\n"
            result += f"ROI区域HSV值:\n"
            result += f"色调(H): {avg_h:.2f}\n"
            result += f"饱和度(S): {avg_s:.2f}\n"
            result += f"亮度(V): {avg_v:.2f}\n\n"
            result += f"多项式回归次数: {self.polynomial_degree}次\n\n"
            result += f"基于色调的预测偏移: {format_offset(h_offset)}\n"
            result += f"基于饱和度的预测偏移: {format_offset(s_offset)}\n\n"
            result += f"平均预测偏移: {format_offset(final_offset)}\n\n"

            self.inference_result_var.set(result)

            if self.reference_time:
                final_time = self.reference_time + timedelta(seconds=final_offset)
                result += f"预测的绝对时间:\n{final_time.strftime('%Y-%m-%d %H:%M:%S')}"
                self.inference_result_var.set(result)

            self.inference_status = "成功"
            self.on_inference_complete(True)

        except Exception as e:
            messagebox.showerror("推理失败", f"推理过程中发生错误: {str(e)}")
            self.inference_result_var.set(f"推理失败: {str(e)}")
            self.inference_status = "失败"
            self.inferred_time = None
            self.update_time_display()
            self.on_inference_complete(False)

    def update_time_display(self):
        if self.inferred_time is not None:
            hours = int(self.inferred_time // 3600)
            minutes = int((self.inferred_time % 3600) // 60)
            secs = int(self.inferred_time % 60)
            self.time_var.set(f"{hours}小时 {minutes}分钟 {secs}秒 ({self.inferred_time:.1f}秒)")
        else:
            self.time_var.set("图像推理未完成或失败")

    def predict_production(self):
        if self.inferred_time is None:
            messagebox.showerror("错误", "请先完成图像推理并获取运行时间")
            self.prediction_status = "失败"
            self.on_prediction_complete(False)
            return

        if (not hasattr(self.production_analyzer, 'model_production') or
            not hasattr(self.production_analyzer, 'model_index') or
            not hasattr(self.production_analyzer, 'poly_features')):
            messagebox.showerror("错误", "请先完成产量指标模型训练")
            self.prediction_status = "失败"
            self.on_prediction_complete(False)
            return

        try:
            # 使用推理得到的时间进行预测
            time_data = np.array([[self.inferred_time]]) / 3600

            # 转换为多项式特征
            time_poly = self.production_analyzer.poly_features.transform(time_data)

            # 预测产量和指标数据
            predicted_production = self.production_analyzer.model_production.predict(time_poly)[0]
            predicted_index = self.production_analyzer.model_index.predict(time_poly)[0]

            # 显示结果
            result = f"产量指标推测结果:\n\n"
            result += f"使用的运行时间: {self.time_var.get()}\n\n"
            result += f"预测产量: {predicted_production:.4f}\n"
            result += f"预测指标数据: {predicted_index:.4f}\n\n"
            result += f"多项式阶数: {self.production_analyzer.degree_var.get()}"

            self.prediction_var.set(result)
            self.prediction_status = "成功"
            self.on_prediction_complete(True)

        except Exception as e:
            messagebox.showerror("推测失败", f"产量指标推测过程中发生错误: {str(e)}")
            self.prediction_var.set(f"推测失败: {str(e)}")
            self.prediction_status = "失败"
            self.on_prediction_complete(False)


def main():
    root = tk.Tk()
    root.title("生产过程分析系统")
    root.geometry("1200x800")

    # 创建主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 创建画布用于绘制流程图
    flow_frame = ttk.LabelFrame(main_frame, text="流程状态", height=100)
    flow_frame.pack(fill=tk.X, pady=5)
    flow_frame.pack_propagate(False)

    canvas_width = 1100
    canvas_height = 80
    canvas = tk.Canvas(flow_frame, width=canvas_width, height=canvas_height)
    canvas.pack(fill=tk.BOTH, expand=True)

    # 绘制流程图元素
    box_width = 180
    box_height = 50
    spacing = 50
    start_x = (canvas_width - (4 * box_width + 3 * spacing)) / 2
    start_y = (canvas_height - box_height) / 2

    rectangles = []
    texts = []

    # 四个流程框
    labels = ["设备运行时间\n模型训练", "产量指标\n模型训练", "图像推理", "产量指标推测"]
    for i, label in enumerate(labels):
        x = start_x + i * (box_width + spacing)
        rect = canvas.create_rectangle(x, start_y, x + box_width, start_y + box_height,
                                      fill="white", outline="black", width=2)
        text = canvas.create_text(x + box_width/2, start_y + box_height/2,
                                 text=label, font=("SimHei", 10), justify=tk.CENTER)
        rectangles.append(rect)
        texts.append(text)

        # 绘制箭头，最后一个不画
        if i < 3:
            arrow_x = x + box_width + spacing/2
            canvas.create_line(arrow_x, start_y + box_height/2,
                              arrow_x + spacing/2, start_y + box_height/2,
                              arrow=tk.LAST, width=2)

    # 顶部按钮区域
    button_frame = ttk.Frame(main_frame, height=50)
    button_frame.pack(fill=tk.X, pady=5)
    button_frame.pack_propagate(False)

    # 内容显示区域
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    # 流程图状态
    global train_status
    train_status = {
        "设备运行时间模型训练": "未完成",
        "产量指标模型训练": "未完成",
        "图像推理": "未完成",
        "产量指标推测": "未完成"
    }

    # 创建分析器实例变量
    hsv_analyzer = None
    production_analyzer = None
    combined_inference_analyzer = None


    # 更新流程图状态的函数
    def update_flow_status():
        for i, key in enumerate(train_status.keys()):
            status = train_status[key]
            if status == "成功":
                canvas.itemconfig(rectangles[i], fill="green")
            elif status == "失败":
                canvas.itemconfig(rectangles[i], fill="red")
            else:
                canvas.itemconfig(rectangles[i], fill="white")

    # 设备运行时间模型训练按钮回调
    def train_hsv_model():
        nonlocal hsv_analyzer

        def on_complete(success):
            train_status["设备运行时间模型训练"] = "成功" if success else "失败"
            update_flow_status()
            if success and hsv_analyzer:
                # 将模型和相关数据传递给推理分析器
                if combined_inference_analyzer:
                    combined_inference_analyzer.models = hsv_analyzer.models
                    combined_inference_analyzer.polynomial_degree = hsv_analyzer.polynomial_degree
                    combined_inference_analyzer.reference_time = hsv_analyzer.reference_time
                    combined_inference_analyzer.time_offsets = hsv_analyzer.time_offsets

        hsv_analyzer = HSVChangeAnalyzer(content_frame, on_complete)

    # 产量指标模型训练按钮回调
    def train_production_model():
        nonlocal production_analyzer

        def on_complete(success):
            train_status["产量指标模型训练"] = "成功" if success else "失败"
            update_flow_status()

        production_analyzer = ProductionModelAnalyzer(content_frame, on_complete)

    # 推理按钮回调
    def show_inference():
        nonlocal combined_inference_analyzer
        nonlocal production_analyzer

        def on_inference_complete(success):
            train_status["图像推理"] = "成功" if success else "失败"
            update_flow_status()

        def on_prediction_complete(success):
            train_status["产量指标推测"] = "成功" if success else "失败"
            update_flow_status()

        # 确保产量分析器已创建
        if not production_analyzer:
            production_analyzer = ProductionModelAnalyzer(content_frame, lambda x: None)

        combined_inference_analyzer = CombinedInferenceAnalyzer(
            content_frame,
            production_analyzer,
            on_inference_complete,
            on_prediction_complete
        )

        # 如果已经训练过HSV模型，传递模型数据
        if hsv_analyzer:
            combined_inference_analyzer.models = hsv_analyzer.models
            combined_inference_analyzer.polynomial_degree = hsv_analyzer.polynomial_degree
            combined_inference_analyzer.reference_time = hsv_analyzer.reference_time
            combined_inference_analyzer.time_offsets = hsv_analyzer.time_offsets
	
    # 创建按钮
    ttk.Button(button_frame, text="设备运行时间模型训练", command=train_hsv_model).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="产量指标模型训练", command=train_production_model).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="推理", command=show_inference).pack(side=tk.LEFT, padx=5)

    # 初始化流程图状态
    update_flow_status()

    root.mainloop()


if __name__ == "__main__":
    main()
