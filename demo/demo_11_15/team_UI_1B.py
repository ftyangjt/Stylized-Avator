import sys
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                            QFileDialog, QLineEdit, QWidget, QMessageBox, QSlider, QGroupBox, QGridLayout, QComboBox,QCheckBox
                            )
from PyQt5.QtGui import QPixmap, QImage 
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PIL import Image
import os
import MY4A, MY4B
import mediapipe as mp
import time
#import mediapipe as mp
#from qt_material import apply_stylesheet

class AvatarStylizer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.pipeline_builded = False

    def initUI(self):

        groupbox_height = 45

        # 设置窗口标题
        self.setWindowTitle('Avatar Stylizer')

        # 设置窗口大小为800x600
        self.resize(800, 600)

        # 标签显示加载的图像
        self.image_label = QLabel('No Image Loaded', self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setScaledContents(False)

        # 按钮：选择图像
        self.load_image_button = QPushButton('Load Image(加载图片)', self)
        self.load_image_button.clicked.connect(self.load_image)
        self.load_image_button.setFixedSize(250,75)  

        # 按钮：构建管道
        self.bulid_pipeline_button = QPushButton('构建管道', self)
        self.bulid_pipeline_button.clicked.connect(self.bulid_pipeline)
        self.bulid_pipeline_button.setFixedSize(250,75) 

        # 按钮：生成风格化图像
        self.generate_button = QPushButton('生成图片，记得构建', self)
        self.generate_button.clicked.connect(self.generate_button_pushed)
        self.generate_button.setFixedSize(250,75)  
  

        # 按钮：保存生成的图像
        self.save_button = QPushButton('Save Stylized Image(保存图片)', self)
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)  # 开始时禁用，生成后启用
        self.save_button.setFixedSize(250,75)



        # 文本框：输入prompt
        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText('Enter prompt...(输入prompt)')
        self.prompt_input.setMinimumHeight(50)
        # 给出默认negative_prompt
        self.negative_prompt_input = QLineEdit(self)
        self.negative_prompt_input.setPlaceholderText('Enter negative prompt...(输入负面prompt)')
        self.negative_prompt_input.setMinimumHeight(50)
        self.negative_prompt_input.setText("ugly, deformed, disfigured, poor details, bad anatomy, Anime")



        # guidance_scale模块
        # 设置guidance_scale滑条
        self.guidance_scale_silder = QSlider(Qt.Horizontal, self)
        self.guidance_scale_silder.setMinimum(1)
        self.guidance_scale_silder.setMaximum(20)
        self.guidance_scale_silder.setValue(8)
        # 输入框
        self.guidance_scale_input = QLineEdit(self)
        self.guidance_scale_input.setText("8")
        self.guidance_scale_input.setFixedWidth(50)
        # 绑定滑条和输入框
        self.guidance_scale_silder.valueChanged.connect(self.update_guidance_scale_input)
        self.guidance_scale_input.textChanged.connect(self.update_guidance_scale_slider)
        # 用groupbox类建立分组框
        self.guidance_scale_group = QGroupBox("guidance_scale(指导程度)", self)
        guidance_scale_layout = QHBoxLayout()
        guidance_scale_layout.addWidget(self.guidance_scale_silder)
        guidance_scale_layout.addWidget(self.guidance_scale_input)
        self.guidance_scale_group.setLayout(guidance_scale_layout) 
        self.guidance_scale_group.setFixedHeight(groupbox_height)     

        # 步数模块
        # 设置步数滑条
        self.steps_silder = QSlider(Qt.Horizontal, self)
        self.steps_silder.setMinimum(5)
        self.steps_silder.setMaximum(100)
        self.steps_silder.setValue(30)
        self.steps_silder.setTickPosition(QSlider.TicksBelow)
        self.steps_silder.setTickInterval(10)
        # 设置步数输入框
        self.steps_input = QLineEdit(self)
        self.steps_input.setText("30")
        self.steps_input.setFixedWidth(50)
        # 绑定滑条和输入框
        self.steps_silder.valueChanged.connect(self.update_steps_input)
        self.steps_input.textChanged.connect(self.update_steps_slider)
        # 用groupbox类建立分组框
        self.steps_group = QGroupBox("Steps(采样步数)", self)
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(self.steps_silder)
        steps_layout.addWidget(self.steps_input)
        self.steps_group.setLayout(steps_layout) 
        self.steps_group.setFixedHeight(groupbox_height)

        # 强度模块
        # 设置强度滑条
        self.strength_silder = QSlider(Qt.Horizontal, self)
        self.strength_silder.setMinimum(0)
        self.strength_silder.setMaximum(100)
        self.strength_silder.setValue(60)
        self.strength_silder.setTickPosition(QSlider.TicksBelow)
        self.strength_silder.setTickInterval(10)
        self.strength_silder.setSingleStep(1)
        # 设置强度输入框
        self.strength_input = QLineEdit(self)
        self.strength_input.setText("0.6")
        self.strength_input.setFixedWidth(50)
        # 绑定滑条和输入框
        self.strength_silder.valueChanged.connect(self.update_strength_input)
        self.strength_input.textChanged.connect(self.update_strength_slider)
        # 用groupbox类建立分组框
        self.strength_group = QGroupBox("Strength(强度)", self)
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_silder)
        strength_layout.addWidget(self.strength_input)
        self.strength_group.setLayout(strength_layout) 
        self.strength_group.setFixedHeight(groupbox_height)

        # 种子模块
        # 种子输入框
        self.seed_input = QLineEdit(self)
        self.seed_input.setPlaceholderText("输入种子")
        self.seed_input.setFixedHeight(30)
        # 随机生成种子按钮
        self.seed_random_botton = QPushButton('随机种子',self)
        self.seed_random_botton.setFixedSize(40,20)
        # 用groupbox类建立分组框
        self.seed_group = QGroupBox("Seed(种子)",self)
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(self.seed_input)
        seed_layout.addWidget(self.seed_random_botton)
        self.seed_group.setLayout(seed_layout)
        self.seed_group.setFixedHeight(50)

        self.control_net_dict = {
            "kind" : None,
        }


        self.model_choice = QComboBox()
        self.model_choice.addItems(["artStyleXXX_v10"])
        #self.control_net_choice.currentTextChanged.connect(self.update_control_net_choice)
        self.model_choice_layout = QVBoxLayout()
        self.model_choice_layout.addWidget(self.model_choice)
        self.model_choice_group = QGroupBox("主模型",self)
        self.model_choice_group.setLayout(self.model_choice_layout)



        self.control_net_map = {
            "无":None,
            "边缘图":"array",
            "深度图":"dpt"
        }

        self.control_net_choice = QComboBox()
        self.control_net_choice.addItems(self.control_net_map.keys())
        self.control_net_choice.currentTextChanged.connect(self.pipeline_changed)
        #self.control_net_choice.currentTextChanged.connect(self.update_control_net_choice)
        self.control_net_choice_layout = QVBoxLayout()
        self.control_net_choice_layout.addWidget(self.control_net_choice)
        self.control_net_choice_group = QGroupBox("控制网模型",self)
        self.control_net_choice_group.setLayout(self.control_net_choice_layout)
        self.control_net_choice_group.setContentsMargins(10,10,10,10)

        
        self.face_option = QCheckBox("面部蒙版")
        self.face_option.stateChanged.connect(self.face_option_change)
        self.face_option.stateChanged.connect(self.pipeline_changed)

        self.face_option_reverse = QCheckBox("反转蒙版")
        self.face_option_reverse.setVisible(False)

        self.face_option_layout = QHBoxLayout()
        self.face_option_layout.addWidget(self.face_option)
        self.face_option_layout.addWidget(self.face_option_reverse)



        # 以下是布局设置
        # 第一行框架
        set_0_layout = QGridLayout()
        set_0_layout_1 = QGridLayout()
        set_0_layout_1.addWidget(self.load_image_button,0,0)
        set_0_layout_1.addWidget(self.bulid_pipeline_button,0,1)
        set_0_layout_1.addWidget(self.generate_button,1,0)
        set_0_layout_1.addWidget(self.save_button,2,0)
        set_0_layout_1.addWidget(self.model_choice_group,3,0)
        set_0_layout_1.addWidget(self.control_net_choice_group,4,0)      
        set_0_layout_1.addLayout(self.face_option_layout,5,0)      
        set_0_layout_1.setSpacing(5)
        set_0_layout_1.setContentsMargins(10,20,10,10)
        set_0_layout.addWidget(self.image_label,0,0,3,3)
        set_0_layout.addLayout(set_0_layout_1,0,3,3,2,alignment=Qt.AlignRight | Qt.AlignTop)

        # 第二行框架
        set_1_layout = QHBoxLayout()
        set_1_layout.addWidget(self.guidance_scale_group)
        set_1_layout.addWidget(self.strength_group)

        # 主布局
        vbox = QGridLayout()
        vbox.addLayout(set_0_layout,1,0)
        vbox.addWidget(self.prompt_input,2,0)
        vbox.addWidget(self.negative_prompt_input,3,0)
        vbox.addLayout(set_1_layout,4,0)
        vbox.addWidget(self.steps_group,5,0)
        vbox.addWidget(self.seed_group,6,0)
        self.setLayout(vbox)

        # 变量存储加载的图像和生成的图像
        self.loaded_image_path = None
        self.generated_image = None





    # 以下是一些辅助函数，用于更新滑条和输入框的值
    # steps滑条和输入框
    def update_steps_input(self):
        self.steps_input.setText(str(self.steps_silder.value()))
    
    def update_steps_slider(self):
        try:
            value = int(self.steps_input.text())
            if 0 <= value <= 1:
                self.steps_silder.setValue(value)
        except:
            pass
    
    # strength滑条和输入框
    def update_strength_input(self):
        self.strength_input.setText(str(self.strength_silder.value()/100))
    
    def update_strength_slider(self):
        try:
            value = int(self.strength_input.text()*100)
            if 0 <= value <= 100:
                self.strength_silder.setValue(value)
        except:
            pass

    # guidance_scale滑条和输入框
    def update_guidance_scale_input(self):
        self.guidance_scale_input.setText(str(self.guidance_scale_silder.value()))
    
    def update_guidance_scale_slider(self):
        try:
            value = int(self.guidance_scale_input.text())
            if 0 <= value <= 20:
                self.guidance_scale_silder.setValue(value)
        except:
            pass


    #面部蒙版选项相关
    def face_option_change(self, state):
        if state == 2:
            self.face_option_reverse.setVisible(True)
        else:
            self.face_option_reverse.setChecked(False)
            self.face_option_reverse.setVisible(False)

    def face_option_judge(self):
        judge = None
        if self.face_option.isChecked() and self.face_option_reverse.isChecked():
            judge = 1
        elif self.face_option.isChecked():
            judge = 0
        else:
            judge = None
        return judge



    # 以下是图片加载、生成和保存的函数
    def load_image(self):
        # 打开文件选择对话框加载图像
        options = QFileDialog.Options()
        file_name_address, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name_address:
            self.loaded_image_path = file_name_address
            pixmap = QPixmap(file_name_address)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("")  # 清除默认标签

    def bulid_pipeline(self, then_generate=False):
        try:
                       
            self.prompt = self.prompt_input.text()
            self.negative_prompt = self.negative_prompt_input.text()
            self.guidance_scale = int(self.guidance_scale_input.text())
            self.steps = self.steps_silder.value()
            self.strength = self.strength_silder.value()/100

            self.face_mask = self.face_option_judge()

            if self.seed_input.text() == "":
                self.seed= -1
            else:
                self.seed = int(self.seed_input.text())
            self.control_net_kind = self.control_net_map[self.control_net_choice.currentText()]

            self.workerthread = WorkerThread(self.loaded_image_path, 
                                  self.prompt, 
                                  self.negative_prompt, 
                                  self.guidance_scale, 
                                  self.steps, 
                                  self.strength, 
                                  self.seed, 
                                  self.control_net_kind,
                                  self.face_mask,
                                  )
            self.workerthread.finished.connect(self.update_pipeline)
            if then_generate==True:
                self.workerthread.finished.connect(self.generate_image)
            self.workerthread.start()
            

        # 捕获异常并显示错误消息
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to generate image: {e}')
            print(e)
    def update_pipeline(self,pipe):
        self.pipeline_builded = True
        self.pipe = pipe   


    def generate_button_pushed(self):
        if self.pipeline_builded == False:
            self.bulid_pipeline(then_generate=True)   
        else:    
            self.generate_image()
        
    def generate_image(self):
        # 检查是否加载了图像
        if not self.loaded_image_path:
            QMessageBox.warning(self, 'Warning', 'Please load an image first.')
            return
        
        # 不需要检查是否输入了prompt
        #prompt = self.prompt_input.text()
        #if not prompt:
            #QMessageBox.warning(self, 'Warning', 'Please enter a prompt.')
            #eturn

        try:

            self.prompt = self.prompt_input.text()
            self.negative_prompt = self.negative_prompt_input.text()
            self.guidance_scale = int(self.guidance_scale_input.text())
            self.steps = self.steps_silder.value()
            self.strength = self.strength_silder.value()/100

            self.face_mask = self.face_option_judge()

            if self.seed_input.text() == "":
                self.seed= -1
            else:
                self.seed = int(self.seed_input.text())
            self.control_net_kind = self.control_net_map[self.control_net_choice.currentText()]

            self.thread = GenerateThread(
                                  self.pipe,
                                  self.loaded_image_path, 
                                  self.prompt, 
                                  self.negative_prompt, 
                                  self.guidance_scale, 
                                  self.steps, 
                                  self.strength, 
                                  self.seed, 
                                  self.control_net_kind,
                                  self.face_mask,
                                  )
            self.thread.finished.connect(self.update_image)
            self.thread.start()

        

        # 捕获异常并显示错误消息
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to generate image: {e}')
            print(e)
    
    #更新图片展示
    def update_image(self,image):
            self.generated_image = image

            # 更新界面显示生成的图像
            qt_image = self.pil_to_qpixmap(image)
            self.image_label.setPixmap(qt_image)
            self.image_label.setScaledContents(False)  # 允许缩放内容
            #self.image_label.setFixedSize(800, 800)  # 设置 QLabel 固定大小
            self.save_button.setEnabled(True)  # 生成成功后允许保存

    #将 PIL 图像转换为 QPixmap,这是为了使大模型结果的IMG形式可以转化为UI能显示的格式
    def pil_to_qpixmap(self, pil_image):
        pil_image = pil_image.convert('RGBA')  # 转换为 RGBA 模式
        data = pil_image.tobytes('raw', 'RGBA')  # 将 PIL 图像数据转换为原始字节数据
        qt_image = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)  # 创建 QImage
        return QPixmap.fromImage(qt_image)  # 转换为 QPixmap

    # 保存生成的图像
    def save_image(self):
        if self.generated_image:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)", options=options)
            if file_name:
                img = self.generated_image
                img.save(file_name)
                QMessageBox.information(self, 'Success', 'Image saved successfully.')

    def pipeline_changed(self):
        self.pipeline_builded = False


#!这是工作线程，分离线程使得生成图片时的即时输出成为可能，也防止UI程序出现“程序未响应”问题
class WorkerThread(QThread):
    finished = pyqtSignal(object)
    def __init__(self, loaded_image_path, prompt, negative_prompt, guidance_scale, steps, strength, seed, control_net_kind, face_mask):
        super().__init__()
        self.loaded_image_path = loaded_image_path
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.strength = strength
        self.seed = seed
        self.control_net_kind = control_net_kind
        self.face_mask = face_mask

    def run(self):

        pipe = MY4B.build_pipeline(self.loaded_image_path, 
                            self.prompt, 
                            self.negative_prompt, 
                            self.guidance_scale, 
                            self.steps, 
                            self.strength, 
                            self.seed, 
                            self.control_net_kind,
                            face_mask = self.face_mask
                            )
        self.finished.emit(pipe)

#!这是生成图片线程
class GenerateThread(QThread):
    finished = pyqtSignal(Image.Image)
    def __init__(self, pipe, loaded_image_path, prompt, negative_prompt, guidance_scale, steps, strength, seed, control_net_kind, face_mask):
        super().__init__()
        self.pipe = pipe
        self.loaded_image_path = loaded_image_path
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.strength = strength
        self.seed = seed
        self.control_net_kind = control_net_kind
        self.face_mask = face_mask

    def run(self):

        new_image = MY4B.generate_image(
                            self.pipe,
                            self.loaded_image_path, 
                            self.prompt, 
                            self.negative_prompt, 
                            self.guidance_scale, 
                            self.steps, 
                            self.strength, 
                            self.seed, 
                            self.control_net_kind,
                            face_mask = self.face_mask
                            )
        
        self.finished.emit(new_image)




# 主函数
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AvatarStylizer()
    ex.show()
    #apply_stylesheet(app, theme='light_blue.xml')
    sys.exit(app.exec_())