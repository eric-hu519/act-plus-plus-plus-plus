from flask import Flask, request
import zlib
import pickle

#用于接收物理机2上传回的视觉追踪结果

app = Flask(__name__)

@app.route('/getinfo', methods=['POST'])
def get_info():
    try:
        # 获取请求中的数据（这是一个字节流）
        compressed_data = request.data

        # 解压缩数据
        decompressed_data = zlib.decompress(compressed_data)

        # 将字节流解码为 Python 对象
        results = pickle.loads(decompressed_data)

        # 打印结果
        print(results)

        return 'Received results successfully'
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'An error occurred', 500

if __name__ == '__main__':
    app.run(host='192.168.31.243', port=1116)