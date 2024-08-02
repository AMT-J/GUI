import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QScreen

class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Data Import and Execute')

        # 获取屏幕大小
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        screen_width = screen.width()
        screen_height = screen.height()

        # 设置窗口大小为屏幕的四分之一
        window_width = screen_width // 2
        window_height = screen_height // 2
        self.resize(window_width, window_height)

        # 将窗口移动到屏幕中心
        self.move((screen_width - window_width) // 2, (screen_height - window_height) // 2)

        self.layout = QVBoxLayout()

        self.label = QLabel('Click the button to import data and run the code.', self)
        self.layout.addWidget(self.label)

        self.button = QPushButton('Import Data and Execute', self)
        self.button.clicked.connect(self.import_and_execute)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

    def import_and_execute(self):
        # 打开文件对话框选择多个数据文件
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Data Files", "", "All Files (*)")
        
        if file_names:
            all_data = ""
            for file_name in file_names:
                try:
                    with open(file_name, 'r') as file:
                        all_data += file.read() + "\n"  # 读取所有文件内容，并添加换行符分隔不同文件的内容
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"An error occurred while reading {file_name}: {e}")
                    return  # 如果读取任何文件失败，终止操作

            # 在这里执行你的代码
            result = self.run_code(all_data)
            
            # 显示结果
            self.label.setText(f'Result: {result}')

    def run_code(self, data):
        # 模拟执行代码
        # 这里你可以放置你的实际处理数据的代码
        # 目前仅返回数据长度作为示例
        return len(data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec())
