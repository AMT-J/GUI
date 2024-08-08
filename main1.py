import sys  # 导入系统模块
from sklearn.linear_model import enet_path  # 从sklearn.linear_model导入enet_path函数
from scipy.sparse import spdiags, eye  # 从scipy.sparse导入spdiags和eye函数
from scipy.sparse.linalg import spsolve  # 从scipy.sparse.linalg导入spsolve函数
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot，并简写为plt
import tensorflow.keras.backend as K  # 导入tensorflow.keras.backend，并简写为K
import copy  # 导入copy模块
import csv  # 导入csv模块
import io  # 导入io模块
import tensorflow as tf  # 导入tensorflow，并简写为tf
import numpy as np  # 导入numpy，并简写为np
from pathlib import Path  # 从pathlib导入Path类
from PyQt5.QtCore import Qt, QThread, pyqtSignal,QMutex, QMutexLocker, QEvent  # 从PyQt5.QtCore导入各种类和函数
from PyQt5.QtGui import QIcon, QTextCursor, QFont,QColor  # 从PyQt5.QtGui导入QIcon、QTextCursor和QFont
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton, QLineEdit, QLabel, QTextEdit, QProgressBar, QSizePolicy, QHBoxLayout, QFileDialog, QSlider, QMessageBox  # 从PyQt5.QtWidgets导入各种类和函数
from tensorflow.keras.layers import Layer  # 从tensorflow.keras.layers导入Layer
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # 从tensorflow.keras.optimizers.schedules导入ExponentialDecay

# 检查 stdout 和 stderr，如果它们为空，则重新定义它们为 io.TextIOWrapper 对象，以防止输出丢失。
if sys.stdout is None:  # 如果sys.stdout为空
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')  # 将sys.stdout重新定义为io.TextIOWrapper
if sys.stderr is None:  # 如果sys.stderr为空
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')  # 将sys.stderr重新定义为io.TextIOWrapper

# 设置 TensorFlow 的日志级别为 'ERROR'，以减少冗长的日志输出
tf.get_logger().setLevel('ERROR')  # 设置TensorFlow的日志级别

# 检查 GPU 是否可用
flag = tf.config.list_physical_devices('GPU')  # 获取可用的物理设备列表

# 定义一个自定义的 Keras 层：SpatialPyramidPooling
class SpatialPyramidPooling(Layer):  # 定义SpatialPyramidPooling类，继承自Layer
    def __init__(self, pool_list, **kwargs):  # 初始化函数
        super().__init__(**kwargs)  # 调用父类的初始化函数
        self.pool_list = pool_list  # 初始化pool_list属性
        self.num_outputs_per_channel = sum([i * i for i in pool_list])  # 计算每个通道的输出数量

    def build(self, input_shape):  # 构建函数
        self.nb_channels = input_shape[-1]  # 获取输入形状的最后一维（通道数）

    def compute_output_shape(self, input_shape):  # 计算输出形状函数
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)  # 返回输出形状

    def call(self, x):  # 调用函数
        input_shape = tf.shape(x)  # 获取输入形状
        num_rows = input_shape[1]  # 获取输入的行数
        num_cols = input_shape[2]  # 获取输入的列数

        outputs = []  # 初始化输出列表
        for num_pool_regions in self.pool_list:  # 遍历pool_list
            x1s = tf.cast(tf.round(tf.linspace(0, num_cols, num_pool_regions + 1)), 'int32')  # 计算列切片位置
            y1s = tf.cast(tf.round(tf.linspace(0, num_rows, num_pool_regions + 1)), 'int32')  # 计算行切片位置

            for ix in range(num_pool_regions):  # 遍历列切片
                for iy in range(num_pool_regions):  # 遍历行切片
                    x_crop = x[:, y1s[iy]:y1s[iy + 1], x1s[ix]:x1s[ix + 1], :]  # 获取切片区域
                    pooled_val = tf.reduce_max(x_crop, axis=(1, 2))  # 对每个区域进行最大池化
                    outputs.append(pooled_val)  # 将池化结果添加到输出列表

        outputs = tf.concat(outputs, axis=-1)  # 将所有池化结果拼接起来
        return outputs  # 返回池化结果

    def get_config(self):  # 获取配置函数
        config = super().get_config()  # 获取父类配置
        config.update({
            'pool_list': self.pool_list,  # 添加pool_list到配置中
        })
        return config  # 返回配置

# 定义一个包含空间金字塔池化层的神经网络模型
def SSPmodel(input_shape):  # 定义SSPmodel函数
    inputs = tf.keras.Input(shape=input_shape)  # 创建输入层
    inputA, inputB = inputs[:, 0, :], inputs[:, 1, :]  # 分割输入为两个部分

    def conv_block(input_tensor):  # 定义卷积块函数
        x = tf.keras.layers.Conv1D(64, kernel_size=7, strides=1, padding='same', kernel_initializer='he_normal')(input_tensor)  # 一维卷积层
        x = tf.keras.layers.BatchNormalization()(x)  # 批量归一化
        x = tf.keras.layers.Activation('relu')(x)  # 激活函数
        return tf.keras.layers.MaxPooling1D(3)(x)  # 最大池化层

    poolA1 = conv_block(inputA)  # 对inputA应用卷积块
    poolB1 = conv_block(inputB)  # 对inputB应用卷积块

    con = tf.keras.layers.concatenate([poolA1, poolB1], axis=2)  # 合并两个输入
    con = tf.expand_dims(con, -1)  # 扩展维度

    conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(con)  # 二维卷积层
    conv1 = tf.keras.layers.BatchNormalization()(conv1)  # 批量归一化
    conv1 = tf.keras.layers.Activation('relu')(conv1)  # 激活函数

    spp = SpatialPyramidPooling([1, 2, 3, 4])(conv1)  # 空间金字塔池化层

    full1 = tf.keras.layers.Dense(1024, activation='relu')(spp)  # 全连接层
    drop1 = tf.keras.layers.Dropout(0.5)(full1)  # Dropout层
    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(drop1)  # 输出层

    model = tf.keras.Model(inputs, outputs)  # 创建模型
    return model  # 返回模型

# 创建目录的函数，如果目录不存在则创建
def mkdir(path):  # 定义mkdir函数
    Path(path).mkdir(parents=True, exist_ok=True)  # 创建目录及其父目录

# 清空目录的函数，删除目录下的所有文件和子目录
def cleardir(path):  # 定义cleardir函数
    for item in Path(path).rglob('*'):  # 遍历目录下的所有文件和子目录
        if item.is_file():  # 如果是文件
            item.unlink()  # 删除文件
        elif item.is_dir():  # 如果是目录
            item.rmdir()  # 删除目录

# 随机打乱数据集和标签
def randomize(dataset, labels):  # 定义randomize函数
    permutation = np.random.permutation(labels.shape[0])  # 生成随机排列
    return dataset[permutation], labels[permutation]  # 返回打乱后的数据集和标签

# 数据读取器，按批次生成数据
def reader(dataX, dataY, batch_size):  # 定义reader函数
    steps = dataX.shape[0] // batch_size  # 计算步数
    if dataX.shape[0] % batch_size != 0:  # 如果数据集不能整除批次大小
        steps += 1  # 增加一步
    while True:  # 无限循环
        for step in range(steps):  # 遍历步数
            start = step * batch_size  # 计算起始索引
            end = (step + 1) * batch_size  # 计算结束索引
            if end > dataX.shape[0]:  # 如果结束索引超出数据集大小
                end = dataX.shape[0]  # 设置为数据集大小
            dataX_batch = dataX[start:end]  # 获取数据批次
            dataY_batch = dataY[start:end]  # 获取标签批次
            yield dataX_batch, dataY_batch  # 生成数据批次和标签批次

# 加载数据的函数
def load_data(datapath):  # 定义load_data函数
    datafileX = datapath / 'training_X.npy'  # 定义数据文件路径
    datafileY = datapath / 'training_Y.npy'  # 定义标签文件路径

    Xtrain0 = np.load(datafileX)  # 加载数据文件
    Ytrain0 = np.load(datafileY)  # 加载标签文件

    XtrainA, YtrainA = process(Xtrain0, Ytrain0)  # 处理数据和标签

    split_idx = int(0.9 * XtrainA.shape[0])  # 计算分割索引
    XXtrain, YYtrain = XtrainA[:split_idx], YtrainA[:split_idx]  # 获取训练集
    XXvalid, YYvalid = XtrainA[split_idx:], YtrainA[split_idx:]  # 获取验证集

    return XXtrain, YYtrain, XXvalid, YYvalid  # 返回训练集和验证集

# 数据处理函数，进行归一化和随机化
def process(Xtrain, Ytrain):  # 定义process函数
    Xtrain /= np.max(Xtrain, axis=3, keepdims=True)  # 对数据进行归一化
    Xtrain = Xtrain.reshape(Xtrain.shape[0] * Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3], 1)  # 重塑数据形状
    Ytrain = Ytrain.reshape(Ytrain.shape[0] * Ytrain.shape[1], 1)  # 重塑标签形状
    Ytrain = tf.keras.utils.to_categorical(Ytrain)  # 将标签转换为one-hot编码
    return randomize(Xtrain, Ytrain)  # 返回随机化后的数据和标签

# 训练线程类
class TrainThread(QThread):  # 定义TrainThread类，继承自QThread
    update_output = pyqtSignal(str)  # 定义字符串信号
    update_progress = pyqtSignal(int)  # 定义整数信号
    update_epoch_metrics = pyqtSignal(float, float)  # 定义浮点数信号
    training_complete = pyqtSignal(dict)  # 定义字典信号

    def __init__(self, batch_size, epochs, parent=None):  # 初始化函数
        super().__init__(parent)  # 调用父类初始化函数
        self.batch_size = batch_size  # 初始化批次大小
        self.epochs = epochs  # 初始化训练轮数
        self._stop_requested = False  # 初始化停止请求标志
        self._mutex = QMutex()  # 初始化互斥锁

    def run(self):  # 运行函数

        device = '/device:GPU:0' if flag else '/device:CPU:0'  # 选择设备（GPU或CPU）

        try:  # 异常处理
            datapath = Path('./data')  # 数据路径
            savepath = Path('./model')  # 模型保存路径
            mkdir(savepath)  # 创建模型保存目录

            Xtrain, Ytrain, Xvalid, Yvalid = load_data(datapath)  # 加载数据

            tf.keras.backend.clear_session()  # 清除Keras会话

            steps_per_epoch = Xtrain.shape[0] // self.batch_size  # 计算每个epoch的步数
            if Xtrain.shape[0] % self.batch_size != 0:  # 如果数据集大小不能整除批次大小
                steps_per_epoch += 1  # 增加一步

            total_steps = steps_per_epoch * self.epochs  # 计算总步数

            decay_steps = total_steps  # 衰减步数
            initial_learning_rate = 0.01  # 初始学习率
            lr_schedule = ExponentialDecay(  # 学习率调度
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=0.95,
                staircase=True
            )

            model = SSPmodel((2, None, 1))  # 创建模型
            model.compile(  # 编译模型
                optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            class ProgressCallback(tf.keras.callbacks.Callback):  # 定义进度回调类
                def __init__(self, stop_requested, epochs, update_epoch_metrics):  # 初始化函数
                    super().__init__()  # 调用父类初始化函数
                    self.stop_requested = stop_requested  # 初始化停止请求标志
                    self.epochs = epochs  # 初始化训练轮数
                    self.update_epoch_metrics = update_epoch_metrics  # 初始化更新epoch指标函数

                def on_epoch_end(self, epoch, logs=None):  # epoch结束时的回调函数
                    if self.stop_requested():  # 如果请求停止
                        self.model.stop_training = True  # 停止训练
                        return  # 返回
                    progress = f"Epoch {epoch + 1}/{self.epochs} - " \
                               f"Accuracy: {logs.get('accuracy') * 100:.2f}% - " \
                               f"Val Accuracy: {logs.get('val_accuracy') * 100:.2f}%"  # 生成进度字符串

                    self.update_output.emit(progress)  # 发出更新输出信号
                    self.update_progress.emit(int(((epoch + 1) / self.epochs) * 100))  # 发出更新进度条信号

            progress_callback = ProgressCallback(  # 创建进度回调实例
                lambda: self._stop_requested,
                self.epochs,
                self.update_epoch_metrics
            )
            progress_callback.update_output = self.update_output  # 设置更新输出信号
            progress_callback.update_progress = self.update_progress  # 设置更新进度条信号

            with tf.device(device):  # 在指定设备上运行
                history = model.fit(  # 训练模型
                    reader(Xtrain, Ytrain, self.batch_size),
                    steps_per_epoch=Xtrain.shape[0] // self.batch_size,
                    epochs=self.epochs,
                    validation_data=reader(Xvalid, Yvalid, self.batch_size),
                    validation_steps=10,
                    callbacks=[progress_callback]
                )

            tf.keras.backend.clear_session()  # 清除Keras会话
            model.save(savepath / 'model.h5')  # 保存模型
            del model  # 删除模型

            self.update_output.emit('训练完成。\n模型已保存。')  # 发出训练完成信号
            self.training_complete.emit(history.history)  # 发出训练完成信号
        except Exception as e:  # 捕获异常
            self.update_output.emit(f'错误: {e}')  # 发出错误信号

    def stop(self):  # 停止函数
        with QMutexLocker(self._mutex):  # 加锁
            self._stop_requested = True  # 设置停止请求标志

# GUI应用程序类，继承自QWidget
class TrainingApp(QWidget):  # 定义TrainingApp类，继承自QWidget
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类初始化函数
        self.initUI()  # 初始化界面
        self.train_thread = None  # 初始化训练线程
        self.detect_device()  # 检测设备（GPU/CPU）
        self.datafile1 = None  # 初始化数据文件路径

    def initUI(self):  # 初始化界面函数
        self.setWindowTitle('拉曼训练应用程序')  # 设置窗口标题

        self.setWindowIcon(QIcon('1.ico'))  # 设置窗口图标

        # 设置整体样式
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
            }
            QLineEdit, QTextEdit, QLabel, QPushButton {
                font-family: 'Segoe UI';
            }
        """)

        main_layout = QVBoxLayout()  # 创建主布局
        form_layout = QGridLayout()  # 创建表单布局
        button_layout = QHBoxLayout()  # 创建按钮布局
        bottom_layout = QVBoxLayout()  # 创建底部布局

        # 创建并设置批次大小输入框
        self.batch_size_input = QLineEdit()  # 创建批次大小输入框
        self.batch_size_input.setPlaceholderText('批次大小')  # 设置占位文本
        self.epochs_input = QLineEdit()  # 创建训练轮数输入框
        self.epochs_input.setPlaceholderText('训练轮数')  # 设置占位文本

        for input_widget in [self.batch_size_input, self.epochs_input]:  # 遍历输入框
            input_widget.setFixedHeight(40)  # 设置固定高度
            input_widget.setStyleSheet("""
                QLineEdit {
                    border: 2px solid #003366; 
                    border-radius: 10px;
                    background-color: #FFFFFF;  
                    padding: 5px;
                    font-size: 18px;
                }
                QLineEdit:hover {
                    border: 2px solid #00BFFF;  
                    background-color: #F0F8FF;  
                }
            """)
            input_widget.setFont(QFont('Segoe UI', 12, QFont.Normal))  # 设置字体

        form_layout.addWidget(self.batch_size_input, 0, 0)  # 将批次大小输入框添加到表单布局
        form_layout.addWidget(self.epochs_input, 0, 1)  # 将训练轮数输入框添加到表单布局

        # 创建并设置按钮
        self.start_button = QPushButton('开始训练')  # 创建开始训练按钮
        self.stop_button = QPushButton('停止训练')  # 创建停止训练按钮
        self.load_file_button = QPushButton("加载文件")  # 创建加载文件按钮
        self.test_button = QPushButton('测试')  # 创建测试按钮

        self.start_button.setIcon(QIcon('start.png'))  # 设置开始训练按钮图标
        self.stop_button.setIcon(QIcon('stop.png'))  # 设置停止训练按钮图标
        self.load_file_button.setIcon(QIcon('load.png'))  # 设置加载文件按钮图标
        self.test_button.setIcon(QIcon('test.png'))  # 设置测试按钮图标

        for button in [self.start_button, self.stop_button, self.test_button, self.load_file_button]:  # 遍历按钮
            button.setFixedHeight(40)  # 设置固定高度
            button.setStyleSheet("""
                QPushButton {
                    border-radius: 10px; 
                    border: 2px solid #003366;
                    background-color: #E0FFFF;
                    padding: 5px;
                    font-size: 18px;
                }
                QPushButton:hover {
                    background-color: #B0E0E6; 
                }
                QPushButton:pressed {
                    background-color: #AFEEEE;
                }
            """)
            button.setFont(QFont('Segoe UI', 12, QFont.Bold))  # 设置字体

        self.start_button.clicked.connect(self.start_training)  # 连接开始训练按钮的点击信号
        self.stop_button.clicked.connect(self.stop_training)  # 连接停止训练按钮的点击信号
        self.load_file_button.clicked.connect(self.load_file)  # 连接加载文件按钮的点击信号
        self.test_button.clicked.connect(self.test_function)  # 连接测试按钮的点击信号

        button_layout.addWidget(self.start_button)  # 将开始训练按钮添加到按钮布局
        button_layout.addWidget(self.stop_button)  # 将停止训练按钮添加到按钮布局
        button_layout.addWidget(self.load_file_button)  # 将加载文件按钮添加到按钮布局
        button_layout.addWidget(self.test_button)  # 将测试按钮添加到按钮布局

        # 创建并设置输出窗口
        self.output_window = QTextEdit()  # 创建输出窗口
        self.output_window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 设置大小策略
        self.output_window.setStyleSheet("""
            QTextEdit {
                border-radius: 10px;
                border: 3px solid #003366;
                background-color: #FFFFFF;
                padding: 10px;
                font-size: 16px;
            }
            QTextEdit:focus {
                border-color: #00BFFF;
                background-color: #F0F8FF;
            }
        """)
        self.output_window.setReadOnly(True)  # 设置为只读
        self.output_window.setFont(QFont('Segoe UI', 12, QFont.Bold))  # 设置字体

        # 创建并设置进度条
        self.progress_bar = QProgressBar()  # 创建进度条
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 设置大小策略
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;  
                border-radius: 10px;        
                background: #FFFFFF;                           
                padding: 1px;        
                text-align: center;       
            }
            QProgressBar::chunk {
                background: #4CAF50;        
                border-radius: 10px;                        
            }
        """)

        self.custom_text = QLabel("设计者：CJLU")  # 创建自定义文本标签
        self.custom_text.setFont(QFont('Comic Sans MS', 10, QFont.Medium, italic=True))  # 设置字体
        self.custom_text.setAlignment(Qt.AlignCenter)  # 设置对齐方式
        self.custom_text.setStyleSheet("""
            QLabel {
                color: #003366;
                margin-top: 20px;
                font-size: 14px;
            }
        """)

        # 添加滑块控件
        self.slider = QSlider(Qt.Horizontal)  # 创建水平滑块
        self.slider.setMinimum(1)  # 设置最小值
        self.slider.setMaximum(100)  # 设置最大值
        self.slider.setValue(50)  # 设置初始值
        self.slider.setTickInterval(1)  # 设置刻度间隔
        self.slider.setTickPosition(QSlider.TicksBelow)  # 设置刻度位置
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #66afe9;
                border: 1px solid #999;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #fff;
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #003366;
                border: 1px solid #444;
                width: 18px;
                height: 18px;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #66afe9;
                border: 1px solid #555;
            }
        """)
        self.slider.valueChanged.connect(self.slider_value_changed) # 连接滑块值改变信号
        

        main_layout.addLayout(form_layout)  # 将表单布局添加到主布局
        main_layout.addWidget(self.slider)  # 将滑块添加到主布局
        main_layout.addLayout(button_layout)  # 将按钮布局添加到主布局
        main_layout.addWidget(self.output_window)  # 将输出窗口添加到主布局
        main_layout.addWidget(self.progress_bar)  # 将进度条添加到主布局
        bottom_layout.addWidget(self.custom_text)  # 将自定义文本标签添加到底部布局
        main_layout.addLayout(bottom_layout)  # 将底部布局添加到主布局

        self.setLayout(main_layout)  # 设置窗口布局

        self.resize_and_center()  # 调整窗口大小并居中显示
        self.installEventFilter(self)  # 安装事件过滤器

    # 调整窗口大小并居中显示
    def resize_and_center(self):  # 定义resize_and_center函数
        screen_size = QApplication.primaryScreen().size()  # 获取屏幕大小
        width, height = screen_size.width() // 2, screen_size.height() // 2  # 计算窗口大小
        self.setGeometry((screen_size.width() - width) // 2, (screen_size.height() - height) // 2, width, height)  # 设置窗口几何

    # 事件过滤器，用于捕捉窗口调整大小事件
    def eventFilter(self, source, event):  # 定义eventFilter函数
        if event.type() == QEvent.Resize:  # 如果事件类型是调整大小
            self.adjust_sizes()  # 调整组件大小
        return super().eventFilter(source, event)  # 调用父类的事件过滤器

    # 调整组件大小
    def adjust_sizes(self):  # 定义adjust_sizes函数
        scale_factor = self.width() / 800  # 计算缩放因子
        font_size = max(12, int(18 * scale_factor))  # 计算字体大小
        button_height = max(40, int(40 * scale_factor))  # 计算按钮高度

        for input_widget in [self.batch_size_input, self.epochs_input]:  # 遍历输入框
            input_widget.setFixedHeight(button_height)  # 设置固定高度
            input_widget.setStyleSheet(f"""
                QLineEdit {{
                    border: 2px solid #003366; 
                    border-radius: 10px;
                    background-color: #FFFFFF;  
                    padding: 5px;
                    font-size: {font_size}px;
                }}
                QLineEdit:hover {{
                    border: 2px solid #00BFFF;  
                    background-color: #F0F8FF;  
                }}
            """)

        for button in [self.start_button, self.stop_button, self.test_button, self.load_file_button]:  # 遍历按钮
            button.setFixedHeight(button_height)  # 设置固定高度
            button.setStyleSheet(f"""
                QPushButton {{
                    border-radius: 10px; 
                    border: 2px solid #003366;
                    background-color: #E0FFFF;
                    padding: 5px;
                    font-size: {font_size}px;
                }}
                QPushButton:hover {{
                    background-color: #B0E0E6; 
                }}
                QPushButton:pressed {{
                    background-color: #AFEEEE;
                }}
            """)

        output_font_size = max(12, int(16 * scale_factor))  # 计算输出窗口字体大小
        self.output_window.setStyleSheet(f"""
            QTextEdit {{
                border-radius: 10px;
                border: 3px solid #003366;
                background-color: #FFFFFF;
                padding: 10px;
                font-size: {output_font_size}px;
            }}
            QTextEdit:focus {{
                border-color: #00BFFF;
                background-color: #F0F8FF;
            }}
        """)

    # 滑块值改变时的回调函数
    def slider_value_changed(self):  # 定义slider_value_changed函数
        value = self.slider.value()  # 获取滑块值
        self.batch_size_input.setText(str(value))  # 设置批次大小输入框的值
        self.update_output_window(f"滑块值: {value}")  # 更新输出窗口

    # 开始训练按钮的回调函数
    def start_training(self):  # 定义start_training函数
        try:  # 异常处理
            batch_size = int(self.batch_size_input.text())  # 获取批次大小
            epochs = int(self.epochs_input.text())  # 获取训练轮数
        except ValueError:  # 捕获异常
            self.update_output_window("错误: 批次大小和训练轮数必须是整数。")  # 更新输出窗口
            QMessageBox.critical(self, "输入错误", "批次大小和训练轮数必须是整数。")  # 弹出错误消息框
            return  # 返回
        self.start_button.setEnabled(False)  # 禁用开始训练按钮
        self.test_button.setEnabled(False)  # 禁用测试按钮
        self.load_file_button.setEnabled(False)  # 禁用加载文件按钮
        self.stop_button.setEnabled(True)  # 启用停止训练按钮

        self.train_thread = TrainThread(batch_size, epochs)  # 创建训练线程
        self.train_thread.update_output.connect(self.update_output_window)  # 连接更新输出窗口信号
        self.train_thread.update_progress.connect(self.update_progress_bar)  # 连接更新进度条信号
        self.train_thread.update_epoch_metrics.connect(self.update_epoch_metrics)  # 连接更新epoch指标信号
        self.train_thread.training_complete.connect(self.on_training_complete)  # 连接训练完成信号

        self.train_thread.start()  # 启动训练线程

    # 停止训练按钮的回调函数
    def stop_training(self):  # 定义stop_training函数
        if self.train_thread:  # 如果训练线程存在
            self.train_thread.stop()  # 停止训练线程
            self.start_button.setEnabled(True)  # 启用开始训练按钮
            self.test_button.setEnabled(True)  # 启用测试按钮
            self.load_file_button.setEnabled(True)  # 启用加载文件按钮
            self.stop_button.setEnabled(False)  # 禁用停止训练按钮
            self.progress_bar.setValue(0)  # 重置进度条

    # 检测设备（GPU/CPU）的函数
    def detect_device(self):  # 定义detect_device函数
        if flag:  # 如果GPU可用
            device_info = "GPU 状态: <font color='green'><b>可用</b></font>"  # 设置设备信息
        else:  # 如果GPU不可用
            device_info = "GPU 状态: <font color='red'><b>不可用</b></font>"  # 设置设备信息
        self.update_output_window(device_info)  # 更新输出窗口
        self.output_window.setTextColor(QColor("black"))

    # 更新输出窗口的函数
    def update_output_window(self, text):  # 定义update_output_window函数
        self.output_window.append(text)  # 在输出窗口中添加文本
        self.output_window.moveCursor(QTextCursor.End)  # 移动光标到文本末尾

    # 更新进度条的函数
    def update_progress_bar(self, value):  # 定义update_progress_bar函数
        self.progress_bar.setValue(value)  # 设置进度条的值

    # 更新每轮训练后的准确率和验证准确率
    def update_epoch_metrics(self, accuracy, val_accuracy):  # 定义update_epoch_metrics函数
        metrics = f"准确率: {accuracy * 100:.2f}%, 验证准确率: {val_accuracy * 100:.2f}%"  # 格式化指标
        self.update_output_window(metrics)  # 更新输出窗口

    # 训练完成后的回调函数
    def on_training_complete(self, history):  # 定义on_training_complete函数
        self.start_button.setEnabled(True)  # 启用开始训练按钮
        self.test_button.setEnabled(True)  # 启用测试按钮
        self.load_file_button.setEnabled(True)  # 启用加载文件按钮
        self.stop_button.setEnabled(False)  # 禁用停止训练按钮

        self.plot_training_history(history)  # 绘制训练历史
        self.progress_bar.setValue(0)  # 重置进度条

    # 绘制训练历史的函数
    def plot_training_history(self, history):  # 定义plot_training_history函数
        QApplication.processEvents()  # 处理所有待处理的事件

        plt.figure(figsize=(12, 6))  # 创建一个图形，设置大小

        plt.subplot(1, 2, 1)  # 创建子图1
        plt.plot(history['loss'], label='Train Loss', color='b')  # 绘制训练损失
        plt.plot(history['val_loss'], label='Validation Loss', color='r')  # 绘制验证损失
        plt.xlabel('Epochs')  # 设置x轴标签
        plt.ylabel('Loss')  # 设置y轴标签
        plt.legend()  # 显示图例
        plt.title('Loss')  # 设置标题

        plt.subplot(1, 2, 2)  # 创建子图2
        plt.plot(history['accuracy'], label='Train Accuracy', color='b')  # 绘制训练准确率
        plt.plot(history['val_accuracy'], label='Validation Accuracy', color='r')  # 绘制验证准确率
        plt.xlabel('Epochs')  # 设置x轴标签
        plt.ylabel('Accuracy')  # 设置y轴标签
        plt.legend()  # 显示图例
        plt.title('Accuracy')  # 设置标题

        plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
        plt.show()  # 显示图像

    # 加载文件的函数
    def load_file(self):  # 定义load_file函数
        options = QFileDialog.Options()  # 创建文件对话框选项
        file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Numpy 文件 (*.npy);;所有文件 (*)", options=options)  # 打开文件对话框
        if file_name:  # 如果选择了文件
            self.datafile1 = file_name  # 设置数据文件路径
            self.update_output_window(f"文件加载完成: {file_name}")  # 更新输出窗口
            QMessageBox.information(self, "文件加载", f"文件加载完成: {file_name}")  # 弹出信息消息框

    # 测试功能的函数
    def test_function(self):  # 定义test_function函数
        if self.datafile1 is None:  # 如果没有加载文件
            self.update_output_window("错误: 未加载文件。请先加载文件。")  # 更新输出窗口
            QMessageBox.critical(self, "错误", "未加载文件。请先加载文件。")  # 弹出错误消息框
            return  # 返回
        datafile1 = self.datafile1  # 获取数据文件路径

        # Whittaker 平滑算法
        def WhittakerSmooth(x: np.ndarray, lamb: float, w: np.ndarray) -> np.ndarray:  # 定义WhittakerSmooth函数
            m = w.shape[0]  # 获取权重数组的长度
            W = spdiags(w, 0, m, m)  # 创建对角稀疏矩阵
            D = eye(m - 1, m, 1) - eye(m - 1, m)  # 创建差分矩阵
            return spsolve((W + lamb * D.transpose() * D), w * x)  # 求解线性方程组

        # airPLS 基线校正算法
        def airPLS(x: np.ndarray, lamb: float = 10, itermax: int = 10) -> np.ndarray:  # 定义airPLS函数
            m = x.shape[0]  # 获取输入数组的长度
            w = np.ones(m)  # 初始化权重数组
            for i in range(itermax):  # 迭代
                z = WhittakerSmooth(x, lamb, w)  # 平滑
                d = x - z  # 计算差异
                if sum(abs(d[d < 0])) < 0.001 * sum(abs(x)):  # 判断收敛
                    break  # 收敛则停止迭代
                w[d < 0] = np.exp(i * d[d < 0] / sum(d[d < 0]))  # 更新权重
                w[d >= 0] = 0  # 非负部分权重置零
            return z  # 返回平滑结果

        # 对矩阵应用 airPLS 基线校正
        def airPLS_MAT(X: np.ndarray, lamb: float = 10, itermax: int = 10) -> np.ndarray:  # 定义airPLS_MAT函数
            B = X.copy()  # 复制输入矩阵
            for i in range(X.shape[0]):  # 遍历每行
                B[i, :] = airPLS(X[i, :], lamb, itermax)  # 应用airPLS
            return X - B  # 返回基线校正结果

        # 对矩阵应用 Whittaker 平滑
        def WhittakerSmooth_MAT(X: np.ndarray, lamb: float = 1) -> np.ndarray:  # 定义WhittakerSmooth_MAT函数
            C = X.copy()  # 复制输入矩阵
            w = np.ones(X.shape[1])  # 初始化权重数组
            for i in range(X.shape[0]):  # 遍历每行
                C[i, :] = WhittakerSmooth(X[i, :], lamb, w)  # 应用Whittaker平滑
            return C  # 返回平滑结果

        _custom_objects = {"SpatialPyramidPooling": SpatialPyramidPooling}  # 自定义对象字典

        datafile0 = './data/database_for_Liquid_and_powder_mixture.npy'  # 定义数据文件路径
        spectrum_pure = np.load(datafile0)  # 加载数据文件

        spectrum_mix = np.load(datafile1)  # 加载混合光谱文件

        csv_reader = csv.reader(open('./data/database_for_Liquid_and_powder_mixture.csv', encoding='utf-8'))  # 打开并读取CSV文件
        DBcoms = [row for row in csv_reader]  # 将CSV文件内容保存到列表中

        num_features = min(spectrum_pure.shape[1], spectrum_mix.shape[1])  # 获取最小特征数

        spectrum_pure = spectrum_pure[:, :num_features]  # 截取纯光谱数据
        spectrum_mix = spectrum_mix[:, :num_features]  # 截取混合光谱数据

        spectrum_pure_sc = spectrum_pure / np.max(spectrum_pure, axis=1, keepdims=True)  # 对纯光谱数据进行归一化
        spectrum_mix_sc = spectrum_mix / np.max(spectrum_mix, axis=1, keepdims=True)  # 对混合光谱数据进行归一化

        X = np.zeros((spectrum_mix_sc.shape[0] * spectrum_pure_sc.shape[0], 2, num_features, 1))  # 初始化X数组

        for p in range(spectrum_mix_sc.shape[0]):  # 遍历混合光谱数据
            for q in range(spectrum_pure_sc.shape[0]):  # 遍历纯光谱数据
                X[int(p * spectrum_pure_sc.shape[0] + q), 0, :, 0] = spectrum_mix_sc[p, :]  # 设置X的第一个通道
                X[int(p * spectrum_pure_sc.shape[0] + q), 1, :, 0] = spectrum_pure_sc[q, :]  # 设置X的第二个通道

        re_model = tf.keras.models.load_model('./model/model.h5', custom_objects=_custom_objects)  # 加载模型
        y = re_model.predict(X)  # 使用模型进行预测

        spectrum_pure = WhittakerSmooth_MAT(spectrum_pure, lamb=1)  # 对纯光谱数据应用Whittaker平滑
        spectrum_pure = airPLS_MAT(spectrum_pure, lamb=10, itermax=10)  # 对纯光谱数据应用airPLS基线校正
        spectrum_mix = WhittakerSmooth_MAT(spectrum_mix, lamb=1)  # 对混合光谱数据应用Whittaker平滑
        spectrum_mix = airPLS_MAT(spectrum_mix, lamb=10, itermax=10)  # 对混合光谱数据应用airPLS基线校正

        results = []  # 初始化结果列表
        pie_data = []  # 初始化饼图数据列表

        for cc in range(spectrum_mix.shape[0]):  # 遍历混合光谱数据
            com = []  # 初始化成分列表
            coms = []  # 初始化成分名称列表
            for ss in range(cc * spectrum_pure.shape[0], (cc + 1) * spectrum_pure.shape[0]):  # 遍历纯光谱数据
                if y[ss, 1] >= 0.5:  # 如果预测值大于等于0.5
                    com.append(ss % spectrum_pure.shape[0])  # 将成分索引添加到列表中

            X = spectrum_pure[com]  # 获取对应成分的纯光谱数据
            coms = [DBcoms[com[h]] for h in range(len(com))]  # 获取对应成分的名称

            _, coefs_lasso, _ = enet_path(X.T, spectrum_mix[cc, :], l1_ratio=0.96, positive=True, fit_intercept=False)  # 使用ElasticNet进行拟合
            ratio = coefs_lasso[:, -1]  # 获取系数
            ratio_sc = copy.deepcopy(ratio)  # 复制系数

            for ss2 in range(ratio.shape[0]):  # 遍历系数
                ratio_sc[ss2] = ratio[ss2] / np.sum(ratio)  # 归一化系数

            result_str = f"第 {cc} 谱图可能包含:"  # 初始化结果字符串
            for comp, ratio in zip(coms, ratio_sc):  # 遍历成分和系数
                result_str += f"\n   - {comp[0]} : {ratio * 100:.2f}%"  # 添加成分和比例到结果字符串
            results.append(result_str)  # 将结果字符串添加到结果列表中

            labels = [comp[0] for comp in coms]  # 获取成分名称列表
            sizes = ratio_sc * 100  # 获取成分比例列表

            filtered_labels = [labels[i] for i in range(len(sizes)) if sizes[i] > 0]  # 过滤比例大于0的成分名称
            filtered_sizes = [sizes[i] for i in range(len(sizes)) if sizes[i] > 0]  # 过滤比例大于0的成分比例

            pie_data.append((filtered_labels, filtered_sizes))  # 将成分名称和比例添加到饼图数据列表中

        num_pie_charts = len(pie_data)  # 获取饼图数量
        fig, axs = plt.subplots(nrows=(num_pie_charts + 1) // 2, ncols=2, figsize=(12, num_pie_charts * 3))  # 创建子图
        axs = axs.flatten()  # 将子图数组展平

        for i, (labels, sizes) in enumerate(pie_data):  # 遍历饼图数据
            wedges, texts, autotexts = axs[i].pie(  # 绘制饼图
                sizes,
                autopct='%1.1f%%',
                startangle=140,
                textprops={'fontsize': 12}
            )
            axs[i].axis('equal')  # 设置饼图为圆形
            axs[i].set_title(f'Spectrum {i}', fontsize=12)  # 设置标题
            axs[i].legend(  # 添加图例
                wedges,
                labels,
                title="Components",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=12
            )

        for j in range(len(pie_data), len(axs)):  # 移除多余的子图
            fig.delaxes(axs[j])

        plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
        plt.show()  # 显示图像

        output_text = "\n\n".join(results)  # 将结果列表连接为字符串
        self.update_output_window(output_text)  # 更新输出窗口
        self.update_output_window("所有饼图已在一个窗口中显示。")  # 更新输出窗口

# 主程序入口
if __name__ == '__main__':  # 主程序入口
    app = QApplication(sys.argv)  # 创建应用程序
    training_app = TrainingApp()  # 创建TrainingApp实例
    training_app.show()  # 显示应用程序
    sys.exit(app.exec_())  # 运行应用程序
