import sys
from sklearn.linear_model import enet_path
from scipy.sparse import spdiags,eye
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt 
import tensorflow.keras.backend as K
import copy
import csv
import io
import tensorflow as tf
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker,Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton, QTextEdit,QMessageBox,
                             QLineEdit, QLabel, QProgressBar, QSizePolicy, QHBoxLayout,QFileDialog,QSlider,QFrame)
from PyQt5.QtGui import QIcon,QTextCursor,QFont,QColor
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
# Set TensorFlow logging level to ERROR to suppress warnings
# Ensure sys.stdout and sys.stderr are not None
if sys.stdout is None:
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
if sys.stderr is None:
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
# Set TensorFlow logging level to ERROR to suppress warnings
tf.get_logger().setLevel('ERROR')
flag=tf.config.list_physical_devices('GPU')

class SpatialPyramidPooling(Layer):
    def __init__(self, pool_list, **kwargs):
        super().__init__(**kwargs)
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i * i for i in pool_list])

    def build(self, input_shape):
        self.nb_channels = input_shape[-1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def call(self, x):
        input_shape = tf.shape(x)
        num_rows = input_shape[1]
        num_cols = input_shape[2]

        # row_length = [tf.cast(num_rows, 'float32') / i for i in self.pool_list]
        # col_length = [tf.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []
        for num_pool_regions in self.pool_list:
            x1s = tf.cast(tf.round(tf.linspace(0, num_cols, num_pool_regions + 1)), 'int32')
            y1s = tf.cast(tf.round(tf.linspace(0, num_rows, num_pool_regions + 1)), 'int32')

            for ix in range(num_pool_regions):
                for iy in range(num_pool_regions):
                    x_crop = x[:, y1s[iy]:y1s[iy+1], x1s[ix]:x1s[ix+1], :]
                    pooled_val = tf.reduce_max(x_crop, axis=(1, 2))
                    outputs.append(pooled_val)

        outputs = tf.concat(outputs, axis=-1)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_list': self.pool_list,
        })
        return config

def SSPmodel(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    inputA, inputB = inputs[:, 0, :], inputs[:, 1, :]

    def conv_block(input_tensor):
        x = tf.keras.layers.Conv1D(64, kernel_size=7, strides=1, padding='same', kernel_initializer='he_normal')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return tf.keras.layers.MaxPooling1D(3)(x)

    poolA1 = conv_block(inputA)
    poolB1 = conv_block(inputB)

    con = tf.keras.layers.concatenate([poolA1, poolB1], axis=2)
    con = tf.expand_dims(con, -1)

    conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(con)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)

    spp = SpatialPyramidPooling([1, 2, 3, 4])(conv1)

    full1 = tf.keras.layers.Dense(1024, activation='relu')(spp)
    drop1 = tf.keras.layers.Dropout(0.5)(full1)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(drop1)

    model = tf.keras.Model(inputs, outputs)
    return model

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def cleardir(path):
    for item in Path(path).rglob('*'):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            item.rmdir()

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    return dataset[permutation], labels[permutation]

def reader(dataX, dataY, batch_size):
    steps = dataX.shape[0] // batch_size
    if dataX.shape[0] % batch_size != 0:
        steps += 1  # Handle cases where dataset is not divisible by batch_size
    while True:
        for step in range(steps):
            start = step * batch_size
            end = (step + 1) * batch_size
            if end > dataX.shape[0]:  # Handle last batch case
                end = dataX.shape[0]
            dataX_batch = dataX[start:end]
            dataY_batch = dataY[start:end]
            yield dataX_batch, dataY_batch

def load_data(datapath):
    datafileX = datapath / 'training_X.npy'
    datafileY = datapath / 'training_Y.npy'

    Xtrain0 = np.load(datafileX)
    Ytrain0 = np.load(datafileY)

    XtrainA, YtrainA = process(Xtrain0, Ytrain0)

    split_idx = int(0.9 * XtrainA.shape[0])
    XXtrain, YYtrain = XtrainA[:split_idx], YtrainA[:split_idx]
    XXvalid, YYvalid = XtrainA[split_idx:], YtrainA[split_idx:]

    return XXtrain, YYtrain, XXvalid, YYvalid

def process(Xtrain, Ytrain):
    Xtrain /= np.max(Xtrain, axis=3, keepdims=True)
    Xtrain = Xtrain.reshape(Xtrain.shape[0]*Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3], 1)
    Ytrain = Ytrain.reshape(Ytrain.shape[0]*Ytrain.shape[1], 1)
    Ytrain = tf.keras.utils.to_categorical(Ytrain)
    return randomize(Xtrain, Ytrain)

class TrainThread(QThread):
    update_output = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    update_epoch_metrics = pyqtSignal(float, float) 
    training_complete = pyqtSignal(dict)

    def __init__(self, batch_size, epochs, parent=None):
        super().__init__(parent)
        self.batch_size = batch_size
        self.epochs = epochs
        self._stop_requested = False
        self._mutex = QMutex()
        

    def run(self):

        device = '/device:GPU:0' if flag else '/device:CPU:0'
        try:
            datapath = Path('./data')
            savepath = Path('./model')
            mkdir(savepath)

            Xtrain, Ytrain, Xvalid, Yvalid = load_data(datapath)
            
            tf.keras.backend.clear_session()

            # Calculate the number of steps per epoch
            steps_per_epoch = Xtrain.shape[0] // self.batch_size
            if Xtrain.shape[0] % self.batch_size != 0:
                steps_per_epoch += 1

            # Calculate the total number of steps
            total_steps = steps_per_epoch * self.epochs

            # Adjust decay_steps based on the total number of steps
            decay_steps = total_steps
            initial_learning_rate = 0.01
            lr_schedule = ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=0.95,
                staircase=True
            )

            model = SSPmodel((2, None, 1))
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, stop_requested, epochs,update_epoch_metrics):
                    super().__init__()
                    self.stop_requested = stop_requested
                    self.epochs = epochs
                    self.update_epoch_metrics = update_epoch_metrics 
                def on_epoch_end(self, epoch, logs=None):
                    if self.stop_requested():
                        self.model.stop_training = True
                        return
                    progress =  f"Epoch {epoch + 1}/{self.epochs} - " \
                                f"Accuracy: {logs.get('accuracy')*100:.2f}% - " \
                                f"Val Accuracy: {logs.get('val_accuracy')*100:.2f}%"
                    
                    self.update_output.emit(progress)
                    self.update_progress.emit(int(((epoch + 1) / self.epochs) * 100))
                    

            progress_callback = ProgressCallback(
                lambda: self._stop_requested,
                self.epochs,
                self.update_epoch_metrics
            )
            progress_callback.update_output = self.update_output
            progress_callback.update_progress = self.update_progress

            with tf.device(device):
                history = model.fit(
                    reader(Xtrain, Ytrain, self.batch_size),
                    steps_per_epoch=Xtrain.shape[0] // self.batch_size,
                    epochs=self.epochs,
                    validation_data=reader(Xvalid, Yvalid, self.batch_size),
                    validation_steps=10,
                    callbacks=[progress_callback]
                )

            tf.keras.backend.clear_session()
            model.save(savepath / 'model.h5')
            del model

            self.update_output.emit('Training complete.\nModel saved.')
            self.training_complete.emit(history.history)
        except Exception as e:
            self.update_output.emit(f'Error: {e}')

        

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop_requested = True

class CustomLineEdit(QLineEdit):
    def __init__(self, min_value, max_value, slider=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.slider = slider

    def keyPressEvent(self, event):
        # Allow control keys and digits
        if event.key() in [Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right, Qt.Key_Enter, Qt.Key_Return, Qt.Key_Tab]:
            super().keyPressEvent(event)
            if not self.text():  # Check if the input is empty
                self.set_slider_to_min()
            return

        if event.text().isdigit():
            current_text = self.text()
            new_text = current_text + event.text()
            
            try:
                if int(new_text) > self.max_value:
                    return  # Ignore input if it exceeds max value
                elif int(new_text) < self.min_value:
                    self.setText(str(self.min_value))  # Set to min value
                    return
            except ValueError:
                pass

            super().keyPressEvent(event)
        else:
            # Ignore any non-digit input
            event.ignore()

    def set_slider_to_min(self):
        if self.slider:
            self.slider.setValue(self.slider.minimum())

class CustomMessageBox(QMessageBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(QIcon('1.ico')) 

class TrainingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.min_batch_size = 1
        self.max_batch_size = 100
        self.min_epochs = 1
        self.max_epochs = 1000

        self.initUI()
        self.train_thread = None
        self.detect_device()
        self.datafile1 = None
        self.progress_bar.setValue(0)

    def initUI(self):
        self.setWindowTitle('Raman App')

        # Set window icon
        self.setWindowIcon(QIcon('1.ico'))

        # Create layout
        main_layout = QVBoxLayout()
        form_layout = QGridLayout()
        button_layout = QHBoxLayout()
        bottom_layout = QVBoxLayout()

        # Add sliders
        self.batch_size_slider = QSlider(Qt.Horizontal)
        self.batch_size_slider.setMinimum(1)
        self.batch_size_slider.setMaximum(100)
        self.batch_size_slider.setValue(50)
        self.batch_size_slider.setTickInterval(10)
        self.batch_size_slider.setSingleStep(10)
        self.batch_size_slider.setPageStep(10)
        self.batch_size_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_size_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #ccc;
                background: #e1e1e1;
                height: 12px;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #4a90e2;
                border: 1px solid #4a90e2;
                height: 12px;
                border-radius: 6px;
            }
            QSlider::add-page:horizontal {
                background: #e1e1e1;
                border: 1px solid #ccc;
                height: 12px;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #333;
                border: 1px solid #666;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -4px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #4a90e2;
                border: 1px solid #333;
            }
        """)

        self.epochs_slider = QSlider(Qt.Horizontal)
        self.epochs_slider.setMinimum(1)
        self.epochs_slider.setMaximum(1000)
        self.epochs_slider.setValue(100)
        self.epochs_slider.setTickInterval(50)
        self.epochs_slider.setSingleStep(50)
        self.epochs_slider.setPageStep(50)
        self.epochs_slider.setTickPosition(QSlider.TicksBelow)
        self.epochs_slider.setStyleSheet(self.batch_size_slider.styleSheet())
        
        # Input fields
        self.batch_size_input = CustomLineEdit(self.min_batch_size, self.max_batch_size, slider=self.batch_size_slider)
        self.batch_size_input.setPlaceholderText(' 批量大小')

    
        self.epochs_input = CustomLineEdit(self.min_epochs, self.max_epochs, slider=self.epochs_slider)
        self.epochs_input.setPlaceholderText(' 训练周期')

        for input in [self.batch_size_input,self.epochs_input]:
            input.setFixedHeight(self.height()//10)
            input.setStyleSheet("""
                QLineEdit {
                    border: 2px solid #003366; 
                    border-radius: 10px;
                    background-color: #F5F5F5;  
                }
                QLineEdit:hover {
                    border: 2px solid #00BFFF;  
                    background-color: #FFFFFF;  
                }
            """)
        
        # Set input font
        font = QFont('Segoe UI', 12, QFont.Normal)
        self.batch_size_input.setFont(font)
        self.epochs_input.setFont(font)

        form_layout.addWidget(self.batch_size_input, 0, 0)
        form_layout.addWidget(self.epochs_input, 0, 1)
        

        # Buttons
        self.start_button = QPushButton('开始训练')
        self.stop_button = QPushButton('停止训练')
        self.load_file_button=QPushButton("加载文件")
        self.test_button = QPushButton('测试')
        # Button Icons
        self.start_button.setIcon(QIcon('start.png'))  
        self.stop_button.setIcon(QIcon('stop.png'))  
        self.load_file_button.setIcon(QIcon('load.png'))  
        self.test_button.setIcon(QIcon('test.png'))  
        

        # Set the width of the buttons
        for button in [self.start_button, self.stop_button, self.test_button,self.load_file_button]:
            button.setFixedHeight(self.height() // 10)
            button.setStyleSheet("""
                    QPushButton { 
                        background-color: #4a90e2;
                        border-radius: 10px; 
                        border: 1px solid gray; 
                    }
                    QPushButton:hover {
                        background-color: #357ABD; 
                    }
                    QPushButton:pressed {
                        background-color: #2a68a3;
                    }
                """)
            
        # Set button font
        font = QFont('Segoe UI', 12, QFont.Bold)
        self.start_button.setFont(font)
        self.stop_button.setFont(font)
        self.load_file_button.setFont(font)
        self.test_button.setFont(font)
        

        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        self.load_file_button.clicked.connect(self.load_file)
        self.test_button.clicked.connect(self.test_function)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.load_file_button)
        button_layout.addWidget(self.test_button)
        

        # Output window
        self.output_window = QTextEdit()
        self.output_window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_window.setStyleSheet("""
                QTextEdit {
                    border-radius: 10px;
                    border: 3px solid black;
                    background-color: transparent;
                    padding: 5px;
                }
            """)
        self.output_window.setReadOnly(True)
        # Set the font for the output window
        output_font = QFont('Segoe UI', 12, QFont.Bold)
        self.output_window.setFont(output_font)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        # Custom text at the bottom
        self.custom_text = QLabel("Designed by CJLU")
        self.custom_text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        font=QFont('Comic Sans MS',10,QFont.Medium,italic=True)
        self.custom_text.setFont(font)


        # Connect signals and slots for batch size
        self.batch_size_slider.valueChanged.connect(self.update_batch_size_input)
        self.batch_size_input.textChanged.connect(self.update_batch_size_slider)

        # Connect signals and slots for epochs
        self.epochs_slider.valueChanged.connect(self.update_epochs_input)
        self.epochs_input.textChanged.connect(self.update_epochs_slider)

        

        # Add sliders and inputs to layout
        sliders_container = QFrame()
        sliders_layout = QHBoxLayout(sliders_container)
        sliders_layout.addWidget(self.batch_size_slider)
        sliders_layout.addWidget(self.epochs_slider)

        # Add widgets to layouts
        main_layout.addLayout(form_layout)
        main_layout.addWidget(sliders_container)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.output_window)
        main_layout.addWidget(self.progress_bar)
        bottom_layout.addWidget(self.custom_text)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
        
        # Resize and center the window
        screen_size = QApplication.primaryScreen().size()
        width, height = screen_size.width() // 2, screen_size.height() // 2
        self.setGeometry((screen_size.width() - width) // 2, (screen_size.height() - height) // 2, width, height)

    def update_batch_size_input(self, value):
        self.batch_size_input.setText(str(value))

    def update_batch_size_slider(self):
        try:
            value = int(self.batch_size_input.text())
            self.batch_size_slider.setValue(value)
        except ValueError:
            pass

    def update_epochs_input(self, value):
        self.epochs_input.setText(str(value))

    def update_epochs_slider(self):
        try:
            value = int(self.epochs_input.text())
            self.epochs_slider.setValue(value)
        except ValueError:
            pass

    def start_training(self):
        try:
            batch_size = int(self.batch_size_input.text())
            epochs = int(self.epochs_input.text())
        except ValueError:
            # Create an error message box
            msg_box = CustomMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Input Error")
            msg_box.setText("Error: Batch Size and Epochs must be integers.")
            msg_box.setInformativeText("Please ensure that both Batch Size and Epochs are valid integer values.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            return
        
        # Disable the Start button, enable the Stop button, and show the loading indicator
        self.start_button.setEnabled(False)
        self.test_button.setEnabled(False)
        self.load_file_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.train_thread = TrainThread(batch_size, epochs)
        self.train_thread.update_output.connect(self.update_output_window)
        self.train_thread.update_progress.connect(self.update_progress_bar)
        self.train_thread.update_epoch_metrics.connect(self.update_epoch_metrics)
        self.train_thread.training_complete.connect(self.on_training_complete)

        self.train_thread.start()

    def stop_training(self):
        if self.train_thread:
            self.train_thread.stop()
            # Re-enable the Start button and disable the Stop button
            self.start_button.setEnabled(True)
            self.test_button.setEnabled(True)
            self.load_file_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            # Reset the progress bar
            self.progress_bar.setValue(0)

    def detect_device(self):
       # Check if GPU is available and set device info accordingly
        if flag:
            device_info = "GPU Status: <font color='green'><b>Available</b></font>"
        else:
            device_info = "GPU Status: <font color='red'><b>Unavailable</b></font>"
        
        # Output the device info to the output window
        self.output_window.append(device_info)
        
        self.output_window.setStyleSheet("""
                QTextEdit {
                    border-radius: 10px;
                    border: 3px solid black;
                    background-color: transparent;
                    padding: 5px;
                }
            """)
        # Set the font for the output window
        output_font = QFont('Segoe UI', 12, QFont.Bold)
        self.output_window.setFont(output_font)

    def update_output_window(self, text):
        self.output_window.append(text)
         # Scroll to the end
        self.output_window.moveCursor(QTextCursor.End)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def update_epoch_metrics(self, accuracy, val_accuracy):
        # Format and append epoch metrics to the output window
        metrics = f"Accuracy: {accuracy*100:.2f}%, Val Accuracy: {val_accuracy*100:.2f}%"
        self.output_window.append(metrics)

    def on_training_complete(self, history):
        # Re-enable the Start button and disable the Stop button
        self.start_button.setEnabled(True)
        self.test_button.setEnabled(True)
        self.load_file_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        self.plot_training_history(history)
        # Reset the progress bar
        self.progress_bar.setValue(0)

    def plot_training_history(self, history):
        QApplication.processEvents()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss', color='b')
        plt.plot(history['val_loss'], label='Validation Loss', color='r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Train Accuracy', color='b')
        plt.plot(history['val_accuracy'], label='Validation Accuracy', color='r')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.tight_layout()
        plt.show()

    def load_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Numpy Files (*.npy)", options=options)
        if file_name:
            self.datafile1 = file_name
            self.update_output_window(f"File loaded: {file_name}")

    def test_function(self):
        if self.datafile1 is None:
            # Create an error message box
            msg_box = CustomMessageBox()  # Use the custom message box with the icon
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("File Loaded Error")
            msg_box.setText("Error: No file loaded.")
            msg_box.setInformativeText("Please load a file first.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            return
        datafile1 = self.datafile1
        
        def WhittakerSmooth(x: np.ndarray, lamb: float , w: np.ndarray) -> np.ndarray:
            m = w.shape[0]
            W = spdiags(w, 0, m, m)
            D = eye(m - 1, m, 1) - eye(m - 1, m)
            return spsolve((W + lamb * D.transpose() * D), w * x)
        
        def airPLS(x: np.ndarray, lamb: float = 10, itermax: int = 10) -> np.ndarray:
            m = x.shape[0]
            w = np.ones(m)
            for i in range(itermax):
                z = WhittakerSmooth(x, lamb, w)
                d = x - z
                if sum(abs(d[d < 0])) < 0.001 * sum(abs(x)):
                    break
                w[d < 0] = np.exp(i * d[d < 0] / sum(d[d < 0]))
                w[d >= 0] = 0
            return z

        def airPLS_MAT(X: np.ndarray, lamb: float=10, itermax: int = 10)-> np.ndarray:
            B = X.copy()
            for i in range(X.shape[0]):
                B[i, :] = airPLS(X[i, :], lamb, itermax)
            return X - B

        def WhittakerSmooth_MAT(X: np.ndarray , lamb: float = 1) -> np.ndarray:
            C = X.copy()
            w = np.ones(X.shape[1])
            for i in range(X.shape[0]):
                C[i, :] = WhittakerSmooth(X[i, :], lamb, w)
            return C

        _custom_objects = {"SpatialPyramidPooling": SpatialPyramidPooling}

        datafile0 = './data/database_for_Liquid_and_powder_mixture.npy'
        spectrum_pure = np.load(datafile0)
        
        #datafile1 = './data/unknown_Liquid_and_powder_mixture.npy'
        spectrum_mix = np.load(datafile1)
        
        csv_reader = csv.reader(open('./data/database_for_Liquid_and_powder_mixture.csv', encoding='utf-8'))
        DBcoms = [row for row in csv_reader]

        num_features = min(spectrum_pure.shape[1], spectrum_mix.shape[1])

        # Trim both datasets to the minimum number of features
        spectrum_pure = spectrum_pure[:, :num_features]
        spectrum_mix = spectrum_mix[:, :num_features]

        # Proceed with normalization
        spectrum_pure_sc = spectrum_pure / np.max(spectrum_pure, axis=1, keepdims=True)
        spectrum_mix_sc = spectrum_mix / np.max(spectrum_mix, axis=1, keepdims=True)

        # Initialize X with the aligned feature size
        X = np.zeros((spectrum_mix_sc.shape[0] * spectrum_pure_sc.shape[0], 2, num_features, 1))

        for p in range(spectrum_mix_sc.shape[0]):
            for q in range(spectrum_pure_sc.shape[0]):
                X[int(p * spectrum_pure_sc.shape[0] + q), 0, :, 0] = spectrum_mix_sc[p, :]
                X[int(p * spectrum_pure_sc.shape[0] + q), 1, :, 0] = spectrum_pure_sc[q, :]

        re_model = tf.keras.models.load_model('./model/model.h5', custom_objects=_custom_objects)
        y = re_model.predict(X)

        spectrum_pure = WhittakerSmooth_MAT(spectrum_pure, lamb=1)
        spectrum_pure = airPLS_MAT(spectrum_pure, lamb=10, itermax=10)
        spectrum_mix = WhittakerSmooth_MAT(spectrum_mix, lamb=1)
        spectrum_mix = airPLS_MAT(spectrum_mix, lamb=10, itermax=10)

        results = []
        pie_data = []

        for cc in range(spectrum_mix.shape[0]):
            com = []
            coms = []
            for ss in range(cc * spectrum_pure.shape[0], (cc + 1) * spectrum_pure.shape[0]):
                if y[ss, 1] >= 0.5:
                    com.append(ss % spectrum_pure.shape[0])

            X = spectrum_pure[com]
            coms = [DBcoms[com[h]] for h in range(len(com))]

            _, coefs_lasso, _ = enet_path(X.T, spectrum_mix[cc, :], l1_ratio=0.96, positive=True, fit_intercept=False)
            ratio = coefs_lasso[:, -1]
            ratio_sc = copy.deepcopy(ratio)
            
            for ss2 in range(ratio.shape[0]):
                ratio_sc[ss2] = ratio[ss2] / np.sum(ratio)
            
            result_str = f"The {cc} spectra may contain:"
            for comp, ratio in zip(coms, ratio_sc):
                result_str += f"\n   - {comp[0]} : {ratio * 100:.2f}%"
            results.append(result_str)

            # Collecting data for pie chart
            labels = [comp[0] for comp in coms]
            sizes = ratio_sc * 100  # Convert to percentage

            # Filter out the components with size 0%
            filtered_labels = [labels[i] for i in range(len(sizes)) if sizes[i] > 0]
            filtered_sizes = [sizes[i] for i in range(len(sizes)) if sizes[i] > 0]

            pie_data.append((filtered_labels, filtered_sizes))

        # Creating subplots
        num_pie_charts = len(pie_data)
        fig, axs = plt.subplots(nrows=(num_pie_charts + 1) // 2, ncols=2, figsize=(12, num_pie_charts * 3))
        axs = axs.flatten()

        for i, (labels, sizes) in enumerate(pie_data):
            wedges, texts, autotexts = axs[i].pie(
                sizes, 
                autopct='%1.1f%%', 
                startangle=140,
                textprops={'fontsize': 12}  # Increase font size for percentages
            )
            axs[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            axs[i].set_title(f'Spectrum {i}', fontsize=12)  # Increase font size for the title
            
            # Set font size for legend
            axs[i].legend(
                wedges, 
                labels, 
                title="Components", 
                loc="center left", 
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=12 # Increase font size for legend
            )

        # Remove empty subplots
        for j in range(len(pie_data), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()

        plt.show()

        # Update the output
        output_text = "\n\n".join(results)
        self.update_output_window(output_text)
        self.update_output_window("All pie charts displayed in a single window.")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    training_app = TrainingApp()
    training_app.show()
    sys.exit(app.exec_())
