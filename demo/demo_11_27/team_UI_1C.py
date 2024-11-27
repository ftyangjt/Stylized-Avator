import sys
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                            QFileDialog, QLineEdit, QWidget, QMessageBox, QSlider, QGroupBox, QGridLayout, QComboBox,QCheckBox,QDialog,QScrollArea,QSpacerItem,QSizePolicy
                            )
from PyQt5.QtGui import QPixmap, QImage 
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PIL import Image
import os
import MY4A, MY4C
import mediapipe as mp
import time
import copy
#import mediapipe as mp
#from qt_material import apply_stylesheet

global_lora_data  = {}
global_mask_data = {}

class AvatarStylizer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.pipeline_builded = False
        self.lora_dict = {}
        self.face_mask= {
            "mask_class" : global_mask_data.get("mask_class",None),
            "datailed_mask_set" : global_mask_data.get("datailed_mask_set",[]),
            "repair_face": global_mask_data.get("repair_face",{"repair_face_bool":None,"repair_strength":0.5})
        } 

    def initUI(self):

        #把下面的改为75可以调大组件
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
        self.bulid_pipeline_button.setFixedSize(150,45) 
        
        # 按钮：
        self.select_lora_button = QPushButton('选择LoRA', self)
        self.select_lora_button.clicked.connect(self.open_lora_dialog)
        self.select_lora_button.setFixedSize(150,45) 

        # 按钮：
        self.select_mask_button = QPushButton('人脸蒙版选项', self)
        self.select_mask_button.clicked.connect(self.open_face_dialog)
        self.select_mask_button.setFixedSize(150,45)
        
        # 按钮：生成风格化图像
        self.generate_button = QPushButton('生成图片', self)
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
        self.steps_silder.setValue(20)
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
        self.model_choice.addItems(["ArtStyle","stable-diffusion-v1-5","majicMIX realistic"])
        self.model_choice.currentTextChanged.connect(self.pipeline_changed)
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



        # 以下是布局设置
        # 第一行框架
        set_0_layout = QGridLayout()
        set_0_layout_1 = QGridLayout()
        set_0_layout_1.addWidget(self.load_image_button,0,0)
        set_0_layout_1.addWidget(self.bulid_pipeline_button,0,1)
        set_0_layout_1.addWidget(self.generate_button,1,0)
        set_0_layout_1.addWidget(self.select_lora_button,1,1)
        set_0_layout_1.addWidget(self.save_button,2,0)
        set_0_layout_1.addWidget(self.select_mask_button,2,1)
        set_0_layout_1.addWidget(self.model_choice_group,3,0)
        set_0_layout_1.addWidget(self.control_net_choice_group,4,0)          
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

    def face_option_judge(self):
        judge = None
        if self.face_option.isChecked() and self.face_option_reverse.isChecked():
            judge = 1
        elif self.face_option.isChecked():
            judge = 0
        else:
            judge = None
        return judge

    #打开人脸蒙版子窗口
    def open_face_dialog(self):
        global global_mask_data
        dialog_mask = MaskDialog(self)
        if dialog_mask.exec_():
                self.pipeline_changed()
                global_mask_data = self.face_mask = dialog_mask.mask_data
                print("用户选择的 mask 数据：", dialog_mask.mask_data)

    #打开LoRA子窗口
    def open_lora_dialog(self):
        global global_lora_data
        """打开选择 LoRA 的对话框"""
        loras = ["森林之光", "fechin", "Pixel Art", "monet_v2", "油画笔触",
                  "沁彩", "墨心", "哈尔的移动城堡", "retrowave  合成波艺术",
                  "Silhouette Synthesis 逆光幻境","华灯初上/Night scene/fantasy city Lora",]
        loras += ["Schematics  黑白底概念图","灵魂卡","苏联海报",]
        dialog = LoraDialog(loras, self)
        if dialog.exec_():
                #copy浅拷贝，因为目标字典中有可变对象，不能完全拷贝，可变对象对应的依然是同一个值
                global_lora_data = dialog.lora_data
                self.lora_dict = dialog.output_data              
                self.pipeline_changed()
                print("用户选择的 LoRA 数据：", dialog.output_data)


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

    #构造和更新pipeline函数
    def bulid_pipeline(self, then_generate=False):
        try:
                       
            self.prompt = self.prompt_input.text()
            self.negative_prompt = self.negative_prompt_input.text()
            self.guidance_scale = int(self.guidance_scale_input.text())
            self.steps = self.steps_silder.value()
            self.strength = self.strength_silder.value()/100
            self.model = self.model_choice.currentText()

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
                                  self.lora_dict,
                                  model = self.model
                                  )
            self.workerthread.finished.connect(self.update_pipeline)
            #因为在生成函数中设置先后顺序，它并不会等到管道构建好了才继续，所以在这里实现先构建再生成
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


    #点击生成按钮判断是否更新了管道
    def generate_button_pushed(self):
        if self.pipeline_builded == False:
            self.bulid_pipeline(then_generate=True)   
        else:    
            self.generate_image()
    #生成图片
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

            if self.seed_input.text() == "":
                self.seed= -1
            else:
                self.seed = int(self.seed_input.text())
            self.control_net_kind = self.control_net_map[self.control_net_choice.currentText()]

            print(self.face_mask)
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



#这个是Lora子窗口
class LoraDialog(QDialog):
    def __init__(self, loras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择 LoRA")
        self.resize(500, 600)
        self.loras = loras  # LoRA名字列表
        # 初始化LoRA
        # 如果全局变量已有数据，则使用；否则初始化默认值
        print(global_lora_data)
        self.lora_data = {
            lora: global_lora_data.get(lora, {"enabled": False, "strength": 1.0})
            for lora in loras
        }
        self.output_data = {}

        # 主布局
        layout = QVBoxLayout(self)

        # 滚动区域
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        #生成 LoRA 控件
        for lora in loras:
            lora_row = QWidget()
            row_layout = QHBoxLayout(lora_row)
            row_layout.setContentsMargins(5, 5, 5, 5)

            # 选择组件（复选框）
            checkbox = QCheckBox()
            checkbox.setChecked(self.lora_data[lora]["enabled"])#这个和下面的一些都是读取上次结果的语句
            checkbox.stateChanged.connect(
                lambda state, l=lora: self.update_enabled(l, state)
            )

            # LoRA 名字
            label = QLabel(lora)
            label.setMinimumWidth(100)

            # 滑块调整强度
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(300)
            slider.setValue(int(self.lora_data[lora]["strength"] * 100))
            slider.setTickInterval(10)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.valueChanged.connect(
                lambda value, l=lora: self.update_strength(l, value / 100)
            )

            # 文本框显示/输入强度
            strength_input = QLineEdit("1.0")
            strength_input.setMaximumWidth(50)
            strength_input.setText(str(self.lora_data[lora]["strength"]))
            strength_input.editingFinished.connect(
                lambda l=lora, input_field=strength_input: self.update_strength_from_input(l, input_field)
            )

            # 滑块与文本框联动
            # !!!小心闭包陷阱，不要把for中的虚拟变量传入lambda
            slider.valueChanged.connect(
                lambda value, input_field=strength_input: input_field.setText(f"{value / 100:.1f}")
            )
            strength_input.textChanged.connect(
                lambda value, input_field=slider: input_field.setValue(int(float(value) * 100))
            )

            row_layout.addWidget(checkbox)
            row_layout.addWidget(label)
            row_layout.addWidget(slider)
            row_layout.addWidget(strength_input)
            scroll_layout.addWidget(lora_row)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # 确定按钮
        confirm_button = QPushButton("确定")
        confirm_button.clicked.connect(self.before_accept)
        layout.addWidget(confirm_button)

    def update_enabled(self, lora_name, state):
        #更新是否启用
        self.lora_data[lora_name]["enabled"] = state == Qt.Checked

    def update_strength(self, lora_name, value):
        #更新强度
        self.lora_data[lora_name]["strength"] = value

    def update_strength_from_input(self, lora_name, input_field):
        #从文本框更新强度
        try:
            value = float(input_field.text())
            if 0 <= value <= 1:
                self.lora_data[lora_name]["strength"] = value
            else:
                raise ValueError
        except ValueError:
            input_field.setText(f"{self.lora_data[lora_name]['strength']:.1f}")

    def before_accept(self):
        for lora_name,data in self.lora_data.items():
            if self.lora_data[lora_name]["enabled"] == True:
                self.output_data[lora_name] = float(f"{self.lora_data[lora_name]['strength']:.1f}")
        print("A")
        self.accept()

#这个是mask子窗口
class MaskDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("蒙版设置")
        self.resize(400, 300)

        self.mask_data = {
            "mask_class" : global_mask_data.get("mask_class",None),
            "datailed_mask_set" : global_mask_data.get("datailed_mask_set",set()),
            "repair_face": global_mask_data.get("repair_face",{"repair_face_bool":None,"repair_strength":0.5})
        }        

        # 主布局
        self.main_layout = QVBoxLayout(self)

        confirm_button = QPushButton("确定")
        confirm_button.clicked.connect(self.before_accept)

        # 固定区域布局
        self.fixed_area_layout = QVBoxLayout()
        self.main_layout.addLayout(self.fixed_area_layout)

        # 下拉框选择蒙版类型
        self.mask_type_label = QLabel("选择蒙版类型:")
        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItems(["无", "粗略蒙版", "精细蒙版"])       
        self.mask_type_combo.currentIndexChanged.connect(self.update_ui)


        # 精细蒙版选项
        self.detailed_mask_options = {}
        for name in ["面部", "衣服", "身体皮肤","饰品"]:
            checkbox = QCheckBox(name)
            checkbox.setVisible(False)
            self.detailed_mask_options[name] = checkbox

        # 修复人脸选项
        self.fix_face_checkbox = QCheckBox("修复人脸")
        self.fix_face_checkbox.setVisible(False)
        self.fix_face_checkbox.stateChanged.connect(self.toggle_face_fix_options)

        # 修复力度选项
        self.fix_strength_label = QLabel("修复力度:")
        self.fix_strength_label.setVisible(False)
        self.fix_strength_slider = QSlider(Qt.Horizontal)
        self.fix_strength_slider.setMinimum(0)
        self.fix_strength_slider.setMaximum(100)
        self.fix_strength_slider.setValue(50)
        self.fix_strength_slider.setVisible(False)
        self.fix_strength_slider.setTickInterval(10)
        self.fix_strength_slider.setTickPosition(QSlider.TicksBelow)
        self.fix_strength_slider.valueChanged.connect(self.sync_text_with_slider)

        self.fix_strength_input = QLineEdit("0.5")
        self.fix_strength_input.setMaximumWidth(50)
        self.fix_strength_input.setVisible(False)
        self.fix_strength_input.editingFinished.connect(self.sync_slider_with_text)

        # 修复力度布局
        self.fix_strength_layout = QHBoxLayout()
        self.fix_strength_layout.addWidget(self.fix_strength_label)
        self.fix_strength_layout.addWidget(self.fix_strength_slider)
        self.fix_strength_layout.addWidget(self.fix_strength_input)

        # 添加到固定区域布局
        self.fixed_area_layout.addWidget(self.mask_type_label)
        self.fixed_area_layout.addWidget(self.mask_type_combo)
        for name,checkbox in self.detailed_mask_options.items():
            self.fixed_area_layout.addWidget(checkbox)
        self.fixed_area_layout.addWidget(self.fix_face_checkbox)
        self.fixed_area_layout.addLayout(self.fix_strength_layout)

        # 添加伸展项防止布局改变
        self.main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.main_layout.addWidget(confirm_button)

        #以下是读取记录
        self.mask_type_combo.setCurrentText(self.mask_data["mask_class"])
        self.detailed_mask_options["面部"].setChecked(3 in self.mask_data["datailed_mask_set"])
        self.detailed_mask_options["衣服"].setChecked(4 in self.mask_data["datailed_mask_set"])
        self.detailed_mask_options["身体皮肤"].setChecked(2 in self.mask_data["datailed_mask_set"])
        self.detailed_mask_options["饰品"].setChecked(5 in self.mask_data["datailed_mask_set"])
        self.fix_face_checkbox.setChecked(bool(self.mask_data["repair_face"]["repair_face_bool"]))

    def update_ui(self):
        """根据蒙版类型更新界面"""
        mask_type = self.mask_type_combo.currentText()
        # 显示精细蒙版的选项
        is_detailed = mask_type == "精细蒙版"
        for name,checkbox in self.detailed_mask_options.items():
            checkbox.setVisible(is_detailed)

        # 显示修复人脸选项
        show_fix_face = mask_type != "无"
        self.fix_face_checkbox.setVisible(show_fix_face)

        # 如果选择了“无”，隐藏所有修复相关内容
        if not show_fix_face:
            self.fix_face_checkbox.setChecked(False)

    def toggle_face_fix_options(self):
        #显示或隐藏修复力度相关选项
        show_fix_options = self.fix_face_checkbox.isChecked()
        self.fix_strength_label.setVisible(show_fix_options)
        self.fix_strength_slider.setVisible(show_fix_options)
        self.fix_strength_input.setVisible(show_fix_options)

    def sync_text_with_slider(self):
        value = round(self.fix_strength_slider.value() / 100, 1)  # 精确到小数点后一位
        self.fix_strength_input.setText(f"{value:.1f}")

    def sync_slider_with_text(self):
        try:
            value = round(float(self.fix_strength_input.text()), 1)  # 精确到小数点后一位
            if 0 <= value <= 1:
                self.fix_strength_slider.setValue(int(value * 100))
            else:
                raise ValueError
        except ValueError:
            self.fix_strength_input.setText(f"{round(self.fix_strength_slider.value() / 100, 1):.1f}")
    
    def before_accept(self):
        
        if self.mask_type_combo.currentText() == "无":
            self.mask_data["mask_class"] = None
            self.accept()
        else:
            self.mask_data["mask_class"] = self.mask_type_combo.currentText()
            
            
        #脸3衣服4身体2
        self.mask_data["datailed_mask_set"] = set()
        if self.detailed_mask_options["面部"].isChecked():
            self.mask_data["datailed_mask_set"].add(3)

        if self.detailed_mask_options["衣服"].isChecked():
            self.mask_data["datailed_mask_set"].add(4)

        if self.detailed_mask_options["身体皮肤"].isChecked():
            self.mask_data["datailed_mask_set"].add(2)
        if self.detailed_mask_options["饰品"].isChecked():
            self.mask_data["datailed_mask_set"].add(5)        

        if self.fix_face_checkbox.isChecked():
            self.mask_data["repair_face"]["repair_face_bool"] = True
            self.mask_data["repair_face"]["repair_strength"] = round(self.fix_strength_slider.value() / 100, 1)
        else:
            self.mask_data["repair_face"]["repair_face_bool"] = False   
        global_mask_data = self.mask_data
        print("A")
        print(self.mask_data)
        self.accept()



#!这是工作线程，分离线程使得生成图片时的即时输出成为可能，也防止UI程序出现“程序未响应”问题
class WorkerThread(QThread):
    finished = pyqtSignal(object)
    def __init__(self, loaded_image_path, prompt, negative_prompt, guidance_scale, steps, strength, seed, control_net_kind, face_mask, lora_dict,model):
        super().__init__()
        self.model = model
        self.loaded_image_path = loaded_image_path
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.strength = strength
        self.seed = seed
        self.control_net_kind = control_net_kind
        self.face_mask = face_mask
        self.lora_dict = lora_dict

    def run(self):

        pipe = MY4C.build_pipeline(
                            
                            self.loaded_image_path, 
                            self.prompt, 
                            self.negative_prompt, 
                            self.guidance_scale, 
                            self.steps, 
                            self.strength, 
                            self.seed, 
                            self.control_net_kind,
                            face_mask = self.face_mask,
                            in_lora_dict = self.lora_dict,
                            model = self.model
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

        new_image = MY4C.generate_image(
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