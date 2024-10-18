# 以下代码完全由AI根据我写的伪代码生成
# 为了运行这个程序，你还需要安装pillow模块

import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import os

# AvatarApp继承自QWiget父类
class AvatarApp(QWidget):
    def __init__(self):
        # 初始化父类的属性
        super().__init__()
        # 调用函数，初始化UI
        self.initUI()

    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle('Avatar Stylizer')

        # 设置窗口大小为800x600
        self.resize(800, 600)

        # 标签显示加载的图像
        self.image_label = QLabel('No Image Loaded', self)
        # 文字居中
        self.image_label.setAlignment(Qt.AlignCenter)

        # 按钮：选择图像
        self.load_image_button = QPushButton('Load Image', self)
        # 如果按键被按下，调用load_image方法
        self.load_image_button.clicked.connect(self.load_image)

        # 文本框：输入prompt
        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText('Enter prompt...')

        # 按钮：生成风格化图像
        self.generate_button = QPushButton('Generate Stylized Image', self)
        self.generate_button.clicked.connect(self.generate_image)

        # 按钮：保存生成的图像
        self.save_button = QPushButton('Save Stylized Image', self)
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)  # 开始时禁用，生成后启用

        # 布局
        # vbox is layout manager
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.load_image_button)
        vbox.addWidget(self.prompt_input)
        vbox.addWidget(self.generate_button)
        vbox.addWidget(self.save_button)

        self.setLayout(vbox)

        # 变量存储加载的图像和生成的图像
        self.loaded_image_path = None
        self.generated_image = None

    def load_image(self):
        # 打开文件选择对话框加载图像
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.loaded_image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio))
            self.image_label.setText("")  # 清除默认标签

    def generate_image(self):
        if not self.loaded_image_path:
            QMessageBox.warning(self, 'Warning', 'Please load an image first.')
            return

        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, 'Warning', 'Please enter a prompt.')
            return

        # 模拟Stable Diffusion模型生成（这里用加载的图像代替）
        try:
            # 假设你已经加载并在后台初始化了SD模型
            new_image = self.run_sd_model(self.loaded_image_path, prompt)

            # 更新界面显示生成的图像
            self.generated_image = new_image
            new_image_pixmap = QPixmap(new_image)
            self.image_label.setPixmap(new_image_pixmap.scaled(600, 600, Qt.KeepAspectRatio))
            self.save_button.setEnabled(True)  # 生成成功后允许保存
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to generate image: {e}')
    
    def run_sd_model(self, image_path, prompt):
        # 这里你需要调用Stable Diffusion模型来处理图像
        # 简单示例：直接返回加载的图像模拟生成（请替换成真实的SD模型调用）
        return image_path  # 返回生成后的图像路径（这是占位符）

    def save_image(self):
        if self.generated_image:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)", options=options)
            if file_name:
                # 保存生成的图像
                img = Image.open(self.generated_image)
                img.save(file_name)
                QMessageBox.information(self, 'Success', 'Image saved successfully.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AvatarApp()
    ex.show()
    sys.exit(app.exec_())
