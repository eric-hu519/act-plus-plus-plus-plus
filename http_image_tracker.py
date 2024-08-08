import gzip
import zlib
import time
import threading
import queue
from flask import Flask, request, Response
import cv2
import numpy as np
import requests
import os
import pickle

from websocket import frame_buffer

app = Flask(__name__)
addr = 'http://172.29.252.92:1110/api'
image_queue = queue.Queue()  # 创建一个队列来存储图像数据
frame_count=0
start_time = None  # 初始化为 None

def display_images():
    while True:
        # 从队列中获取图像数据
        img = image_queue.get()
        if img is None:
            break
        # 实时显示图像
        cv2.imshow('Received Image', img)
        cv2.waitKey(1)  # 1ms 延迟以允许图像刷新

# 创建并启动显示图像的线程
display_thread = threading.Thread(target=display_images)
display_thread.start()        

@app.route('/upload', methods=['POST'])
def upload_image():
    global frame_count, start_time

    # 获取图像数据
    compressed_data = request.data
    # 解压图像数据
    image_data = zlib.decompress(compressed_data)
    # 将字节数据转换为 NumPy 数组
    nparr = np.frombuffer(image_data, np.uint8)

    # 解码图像
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 如果是第一张图像，初始化 start_time
    if frame_count == 0:
        start_time = time.time()

    # 增加帧计数
    frame_count += 1

    # 计算并打印帧率
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
        print(f"Frame rate: {fps:.2f} FPS")
    
    # 将解码后的图像放入队列
    image_queue.put(img)

    # 将图像发送到追踪程序
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    req = requests.post(addr + '/track', files={'image': ('image.jpg', img_bytes)})
    print(req.json())  # 打印追踪结果

    # 返回响应
    return Response(status=200)

if __name__ == '__main__':

    # TRACKING START
    req = requests.request("POST", addr + '/ctl', data={'ctl':1})
    print(req.json()) # {'msg': 'tracker init.'}

    # INIT TRACKING with NATUAL LANGUAGE==
    req = requests.request("POST", addr + '/cmd', data={'nlp': 'red straw on the black table'})
    print(req.json()) # {'msg': 'init success. the (nlp) we get: 'red straw on the black table'}

    try:
        app.run(host='127.0.0.1', port=1110)
    finally:
        # 结束显示线程
        image_queue.put(None)
        display_thread.join()
        cv2.destroyAllWindows()

