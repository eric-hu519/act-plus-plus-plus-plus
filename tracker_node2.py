from platform import node
import threading
import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
import log
import requests
from requests.exceptions import Timeout,ConnectionError, RequestException
import os
import pickle
import time
import zlib
from flask import Flask, request, Response,jsonify
import sys
import json

#   参与联调，在接收llm_node的nlp信息之前挂起
#  "tracker_node.py"是布置在物理机1上的视觉追踪接口，跟物理机2的视觉追踪http_tracker_class4进行交互：
#     接收robot_node发送过来的rpc请求和图片输出，并在接收到图片后进行视觉追踪，最后输出object的坐标和视频流平均帧数
#     接收llm_node的rpc消息，即llm_response为nlp
#     这个文件是接收nodesdk消息，并调用http_image_tracker_class.py中的接受图像和处理图像函数
#     本文件是在本地进行图像处理，不经过云端
#     track_node 与 robot_node 是采用局域网内的http图像传输

#  先运行服务器的tracker_divide.py后，再运行tracker_node.py
# 物理机1给物理机2发送图片的url为：http://192.168.31.109:1115/upload （在物理机2的tracker_class4里）
# 物理机1给物理机2发送nlp的url为： http://192.168.31.109:1116/vision
# 物理机1从物理机2得到追踪的result为： 'host':'192.168.31.243', 'port':1116

NODE_HUB_HOST = "127.0.0.1"
ASR_TOPIC = "/eai/system/voice"
LLM_TOPIC = "/eai/system/command"
BOT_TOPIC = "/eai/system/robot"
TRACKER_TOPIC = "/eai/system/visualtrack/position"
MODEL_DB = "robot_task.json"
APP_NAME = "visual_node"
NODE_ID = "eai.system.track"
IMAGE_URL = "http://192.168.31.109:1115"



class NodeSDKAPI:
    def __init__(self):
        self.task_event = threading.Event()
        self.node_id = 'eai.system.track'
        self._client = nodeclient.NodeClient(self.node_id, node_hub_host=NODE_HUB_HOST)
        _logger = log.init(APP_NAME, verbose=False)
        self._client.register_method("VisualTrack", self._on_start_call)
        self._client.set_on_message(self._on_message)#ljy 处理robotnode push到Bot_Topic的"Task Complete"消息
        self.subscriber(BOT_TOPIC)
        
        #在NodesSDKAPI内部创建Flask应用，并定义了一个路由 /getinfo 来处理从物理机2返回的结果。
        self.app = Flask(__name__)
        self.app.add_url_rule('/getinfo', 'get_info', self.get_info, methods=['POST'])
        #lask 应用是在 NodeSDKAPI 类的实例中创建的，它的生命周期与 NodeSDKAPI 实例相同。
        #这意味着只要 NodeSDKAPI 实例存在，Flask 应用就存在，无论你的脚本是作为主程序运行还是被其他脚本导入作为模块使用

        # Start a thread to monitor the robot_node status
        self._monitor_thread = threading.Thread(target=self._on_message)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        self.llm_response = None    #llm_node传来的nlp消息
        self.Taskcomplete = None    #robotnode传来的任务结束消息



    def get_info(self):
        try:
            # 获取请求中的数据（这是一个字节流）
            compressed_data = request.data
            # 打印原始数据
            #print(f"Received data: {compressed_data}")
            
            # 检查数据是否是字节流
            if not isinstance(compressed_data, bytes):
                print(f"Data is not bytes: {type(compressed_data)}")
                return 'Data is not bytes', 400
            
            # 检查是否接收到了数据
            #if not compressed_data:
             #   raise Exception("No data received")

            # 尝试解压缩数据
            try:
                decompressed_data = zlib.decompress(compressed_data)
            except zlib.error as e:
                print(f"Failed to decompress data: {e}")
                return 'Failed to decompress data', 500

            self.push(decompressed_data)  #将得到的结果字节流pub出去
            # 将字节流转换为字符串
            results_str = decompressed_data.decode("utf-8")
            # 将字符串转换为字典
            results = json.loads(results_str)

            # 打印结果
            print(results)
            results_str = str(results)
            

            return 'Received results successfully'
       
        except Exception as e:
            print(f"An error occurred: {e}")
  
            return 'An error occurred', 500
    


    def push(self, msg: str) -> None:
        rc = self._client.publish(TRACKER_TOPIC, payload=msg, payload_type=nodesdk.ContentType.PB,
                                    qos=nodesdk.Qos.MB_QOS1)
        if not rc:
            log.logger.error(f"Publish error")
    
    def push_nlp_to_visiontracker(self, nlp):#将nlp消息发送给物理机2
        url = 'http://192.168.31.109:1115/vision'  # 物理机2的地址
        headers = {'Content-Type': 'application/json'}  # 设置HTTP头部为JSON
        data = {'nlp': nlp}  # 将nlp消息包装成一个字典
        try:
            response = requests.post(url, headers=headers, json=data)  # 发送POST请求
            response.raise_for_status()
            print('Successfully pushed the message to the other machine.')
        except (ConnectionError, RequestException) as e:
            print(f'Failed to push the message to the other machine with error: {e}')

    #订阅llm_node的rpc消息,接收nlp_response
    def _on_start_call(self, req_id: int, content: bytes, content_type: nodesdk.ContentType, source: str, timeout:int) -> None:
        log.logger.info(f"req_id: {req_id}, content: {content}, content_type: {content_type}, source: {source}, timeout: {timeout}")
        self.llm_response = content.decode("utf-8") #decoded as string
        print(self.llm_response)
        if self.llm_response is not None:
            #将接收到的nlp的rpc消息发送给物理机2的tracker_class.py进行处理：
           
            log.logger.info(f"llm_response: {self.llm_response}")
            self.push_nlp_to_visiontracker(self.llm_response)  # 将nlp消息推送到物理机2
        else:
            log.logger.error("llm_response is None")
        
    def subscriber(self,node_name: str):
        err,topic_filter = self._client.subscribe(
            [nodesdk.TopicFilter(topic_filter=node_name, qos=nodesdk.Qos.MB_QOS1)]
        )#ljy
        if err!= nodesdk.ClientErr.OK:
            log.logger.error(f"Subscribe error: {err}")
            #print("Subscribe error: {err}")
            return#ljy
        log.logger.info(f"Subscribe Success")

    def _on_message(self,msg):
        #get inference status from bot topic
        if msg.topic == BOT_TOPIC:
            bot_status = msg.payload.decode('utf-8')
            assert bot_status in ['Model not found', 'Task Begin', 'Error in running inference', 'Task completed' ], f"Invalid bot status: {bot_status}"
            log.logger.info(f"Receive bot status: {bot_status}")
            if bot_status == 'Task completed':  #ljy:当监测到robot_node发布的消息为"Task completed"时，重启本tracker_node节点
                #self.reset()
                taskcomplete = bot_status
                               
                #taskcomplete = bot_status.encode('utf-8')  # 将字符串转换为字节
                self.push_over_to_visiontracker(taskcomplete)
                print("successful send Task completed to another machine") 

    
        
        """    
    def reset(self): #目前会多此重新初始化，且会导致重复订阅，且阻碍原来的程序运行！！！！
        #log.logger.info(f"robot_node is completed : Resetting tracker node")
        print("robot_node is completed : Resetting tracker node")
        self.llm_response = None    #llm_node传来的nlp消息
        self.Taskcomplete = None    #robotnode传来的任务结束消息    

        print(111)"""

    
    def push_over_to_visiontracker(self, msg):  #将robotnode的进程结束消息发送给物理机2，以便物理机2重启
        max_retries = 3
        attempt = 0
        #print(1111)
        url = 'http://192.168.31.109:1115/taskcomplete'  # 物理机2的地址和taskcomplete路由
        headers = {'Content-Type': 'application/json'}  # 设置HTTP头部为JSON
        data = {'msg': msg}  # 将msg消息包装成一个字典
        while attempt < max_retries:
            try:
                response = requests.post(url, headers=headers, json=data,timeout=5)  # 发送POST请求
                response.raise_for_status()
                print('Successfully pushed Taskcomplete to the other machine.')
                break
            except (Timeout, RequestException) as e:
                attempt +=1
                print(f'Failed to push Taskcomplete to the other machine with error: {e}')


           
if __name__ == "__main__":
    try:
        api = NodeSDKAPI()
        
        #创建了一个新的线程来运行 Flask 应用，这样 Flask 应用和主程序可以同时运行
        threading.Thread(target=api.app.run, kwargs={'host':'192.168.31.243', 'port':1116}).start()
        while True: 
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


