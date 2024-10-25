

import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit, QWidget, QMessageBox, QSlider, QGroupBox
from PyQt5.QtGui import QPixmap, QImage 
from PyQt5.QtCore import Qt
from PIL import Image
import os
import MY3A

class AvatarApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.setStyleSheet("""

    QSlider::handle {
        background: #fff; /* 滑块颜色 */
        border: 2px solid #020447; /* 滑块边框 */
        border-radius: 10px; /* 滑块圆角 */
    }
    QSlider::handle:horizontal:hover {
        background: #45a049; /* 悬停效果 */
    }

""")





        self.setWindowTitle('Avatar Stylizer')

        # 设置窗口大小为800x600
        self.resize(800, 600)

        # 标签显示加载的图像
        self.image_label = QLabel('No Image Loaded', self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(512,512)
        

        # 按钮：选择图像
        self.load_image_button = QPushButton('Load Image(加载图片)', self)
        self.load_image_button.clicked.connect(self.load_image)
        self.load_image_button.setFixedSize(200,75)

        # 文本框：输入prompt
        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText('Enter prompt...(输入prompt)')
        self.prompt_input.setMinimumHeight(50)
        #
        self.negative_prompt_input = QLineEdit(self)
        self.negative_prompt_input.setPlaceholderText('Enter negative prompt...(输入负面prompt)')
        self.negative_prompt_input.setMinimumHeight(50)
        self.negative_prompt_input.setText("ugly, deformed, disfigured, poor details, bad anatomy")

        # 按钮：生成风格化图像
        self.generate_button = QPushButton('Generate Stylized Image(生成图片)', self)
        self.generate_button.clicked.connect(self.generate_image)
        self.generate_button.setFixedSize(200,75)        

        # 按钮：保存生成的图像
        self.save_button = QPushButton('Save Stylized Image(保存图片)', self)
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)  # 开始时禁用，生成后启用
        self.save_button.setFixedSize(200,75)
               
        #guidance_scale滑条
        self.guidance_scale_silder = QSlider(Qt.Horizontal, self)
        self.guidance_scale_silder.setMinimum(1)
        self.guidance_scale_silder.setMaximum(20)
        self.guidance_scale_silder.setValue(8)
        #
        self.guidance_scale_input = QLineEdit(self)
        self.guidance_scale_input.setText("8")
        self.guidance_scale_input.setFixedWidth(50)
        #
        self.guidance_scale_silder.valueChanged.connect(self.update_guidance_scale_input)
        self.guidance_scale_input.textChanged.connect(self.update_guidance_scale_slider)

        #步数滑条
        self.steps_silder = QSlider(Qt.Horizontal, self)
        self.steps_silder.setMinimum(5)
        self.steps_silder.setMaximum(100)
        self.steps_silder.setValue(30)
        self.steps_silder.setTickPosition(QSlider.TicksBelow)
        self.steps_silder.setTickInterval(10)
        #
        self.steps_input = QLineEdit(self)
        self.steps_input.setText("30")
        self.steps_input.setFixedWidth(50)
        #
        self.steps_silder.valueChanged.connect(self.update_steps_input)
        self.steps_input.textChanged.connect(self.update_steps_slider)

        #强度滑条,强度应该是小数，但滑条的值必须是整数，所以在绑定函数中进行了除100处理
        self.strength_silder = QSlider(Qt.Horizontal, self)
        self.strength_silder.setMinimum(0)
        self.strength_silder.setMaximum(100)
        self.strength_silder.setValue(60)
        self.strength_silder.setTickPosition(QSlider.TicksBelow)
        self.strength_silder.setTickInterval(10)
        self.strength_silder.setSingleStep(1)
        #
        self.strength_input = QLineEdit(self)
        self.strength_input.setText("0.6")
        self.strength_input.setFixedWidth(50)
        #
        self.strength_silder.valueChanged.connect(self.update_strength_input)
        self.strength_input.textChanged.connect(self.update_strength_slider)


        #种子模块
        self.seed_input = QLineEdit(self)
        self.seed_input.setPlaceholderText("输入种子")
        self.seed_random_botton = QPushButton('随机种子',self)
        self.seed_random_botton.setFixedSize(40,20)




        # 布局
        vbox = QVBoxLayout()
        #
        self.guidance_scale_group = QGroupBox("guidance_scale(指导程度)", self)
        guidance_scale_layout = QHBoxLayout()
        guidance_scale_layout.addWidget(self.guidance_scale_silder)
        guidance_scale_layout.addWidget(self.guidance_scale_input)
        self.guidance_scale_group.setLayout(guidance_scale_layout) 
        self.guidance_scale_group.setFixedHeight(45)          
        #steps布局,使用groupbox类建立分组框
        self.steps_group = QGroupBox("Steps(采样步数)", self)
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(self.steps_silder)
        steps_layout.addWidget(self.steps_input)
        self.steps_group.setLayout(steps_layout) 
        self.steps_group.setFixedHeight(45)
        #强度布局
        self.strength_group = QGroupBox("Strength(强度)", self)
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_silder)
        strength_layout.addWidget(self.strength_input)
        self.strength_group.setLayout(strength_layout) 
        self.strength_group.setFixedHeight(45)
        #种子布局
        self.seed_group = QGroupBox("Seed(种子)",self)
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(self.seed_input)
        seed_layout.addWidget(self.seed_random_botton)
        self.seed_group.setLayout(seed_layout)
        self.seed_group.setFixedHeight(45)




        #图片行框架，包含重要按钮
        set_0_layout = QHBoxLayout()
        set_0_layout_1 = QVBoxLayout()
        set_0_layout_1.addWidget(self.load_image_button)
        set_0_layout_1.addWidget(self.generate_button)
        set_0_layout_1.addWidget(self.save_button)
        set_0_layout_1.setSpacing(0)
        set_0_layout.addWidget(self.image_label)
        set_0_layout.addLayout(set_0_layout_1)

        #第一行框架
        set_1_layout = QHBoxLayout()
        set_1_layout.addWidget(self.guidance_scale_group)
        set_1_layout.addWidget(self.strength_group)




        vbox.addLayout(set_0_layout)
        vbox.addWidget(self.prompt_input)
        vbox.addWidget(self.negative_prompt_input)
        vbox.addLayout(set_1_layout)
        vbox.addWidget(self.steps_group)
        vbox.addWidget(self.seed_group)
        self.setLayout(vbox)


        # 变量存储加载的图像和生成的图像
        self.loaded_image_path = None
        self.generated_image = None

    #steps滑条和输入框，双向绑定
    def update_steps_input(self):
        self.steps_input.setText(str(self.steps_silder.value()))
    
    def update_steps_slider(self):
        try:
            value = int(self.steps_input.text())
            if 0 <= value <= 1:
                self.steps_silder.setValue(value)
        except:
            pass
    
    #strength滑条和输入框
    def update_strength_input(self):
        self.strength_input.setText(str(self.strength_silder.value()/100))
    
    def update_strength_slider(self):
        try:
            value = int(self.strength_input.text()*100)
            if 0 <= value <= 100:
                self.strength_silder.setValue(value)
        except:
            pass

    #guidance_scale滑条和输入框
    def update_guidance_scale_input(self):
        self.guidance_scale_input.setText(str(self.guidance_scale_silder.value()))
    
    def update_guidance_scale_slider(self):
        try:
            value = int(self.guidance_scale_input.text())
            if 0 <= value <= 20:
                self.guidance_scale_silder.setValue(value)
        except:
            pass














    def load_image(self):
        # 打开文件选择对话框加载图像
        options = QFileDialog.Options()
        file_name_address, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name_address:
            self.loaded_image_path = file_name_address
            pixmap = QPixmap(file_name_address)
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
            self.prompt = self.prompt_input.text()
            self.negative_prompt = self.negative_prompt_input.text()
            self.guidance_scale = int(self.guidance_scale_input.text())
            self.steps = self.steps_silder.value()
            self.strength = self.strength_silder.value()/100
            self.seed = int(self.seed_input.text())
            new_image = MY3A.test(self.loaded_image_path, self.prompt, self.negative_prompt, self.guidance_scale, self.steps, self.strength, self.seed)
            self.generated_image = new_image

            # 更新界面显示生成的图像\
            qt_image = self.pil_to_qpixmap(new_image)
            self.image_label.setPixmap(qt_image)
            self.image_label.setScaledContents(False)  # 允许缩放内容
            #self.image_label.setFixedSize(800, 800)  # 设置 QLabel 固定大小

            self.save_button.setEnabled(True)  # 生成成功后允许保存
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to generate image: {e}')
    



    #将 PIL 图像转换为 QPixmap,这是为了使大模型结果的IMG形式可以转化为UI能显示的格式
    def pil_to_qpixmap(self, pil_image):

        pil_image = pil_image.convert('RGBA')  # 转换为 RGBA 模式
        data = pil_image.tobytes('raw', 'RGBA')  # 将 PIL 图像数据转换为原始字节数据
        qt_image = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)  # 创建 QImage
        return QPixmap.fromImage(qt_image)  # 转换为 QPixmap




    def save_image(self):
        if self.generated_image:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)", options=options)
            if file_name:
                # 保存生成的图像
                img = self.generated_image
                img.save(file_name)
                QMessageBox.information(self, 'Success', 'Image saved successfully.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AvatarApp()
    ex.show()
    sys.exit(app.exec_())