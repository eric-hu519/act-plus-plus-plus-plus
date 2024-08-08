import gzip
import zlib
import time
import threading
import queue
from flask import Flask, request, Response
import cv2
import numpy as np

app = Flask(__name__)

frame_count = 0
start_time = None  # 初始化为 None
image_queue = queue.Queue()  # 创建一个队列来存储图像数据

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

    # 返回响应
    return Response(status=200)

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=1115)
    finally:
        # 结束显示线程
        image_queue.put(None)
        display_thread.join()
        cv2.destroyAllWindows()