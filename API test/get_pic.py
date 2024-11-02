import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def download_image(url):
    # 发送HTTP GET请求
    response = requests.get(url)
    
    # 检查请求是否成功
    if response.status_code == 200:
        # 使用BytesIO将图片数据加载到内存中
        img_data = BytesIO(response.content)
        # 打开图片
        img = Image.open(img_data)
        # 显示图片
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
    else:
        print(f"无法下载图片，状态码: {response.status_code}")

# 示例用法
image_url = 'http://127.0.0.1:5000/image'  # 替换为实际的图片URL
download_image(image_url)