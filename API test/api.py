from flask import Flask, send_file
from datetime import datetime
import os

app = Flask(__name__)

@app.route('/time')
def get_current_time():
    return {'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

@app.route('/image')
def get_image():
    # 获取当前文件的目录
    current_directory = os.path.dirname(__file__)

    # 获取上一个目录
    parent_directory = os.path.abspath(os.path.join(current_directory, '..'))

    # 构建目标文件的路径
    image_path = os.path.join(parent_directory, 'test_image.jpg')

    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()