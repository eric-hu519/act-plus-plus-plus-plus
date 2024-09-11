
import array
from platform import node
from socket import timeout

from regex import F
import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
from ACT_API import ACT_API
import threading
import log
import cv2
import requests
import time
import gzip
import zlib
from sympy import Array #ljy
from requests.exceptions import Timeout, RequestException

NODE_HUB_HOST = "127.0.0.1"
ASR_TOPIC = "/eai/system/voice"
LLM_TOPIC = "/eai/system/command"
BOT_TOPIC = "/eai/system/robot"
MODEL_DB = "/home/mamager/interbotix_ws/src/aloha/act-plus-plus/robot_task.json"
APP_NAME = "robot_node"
#IMAGE_URL = "http://127.0.0.1:1115/upload"
IMAGE_URL = "http://192.168.31.109:1115/upload"
class NodeSDKAPI:
    def __init__(self):
        self.task_event = threading.Event()
        self.node_id = 'eai.system.robot'
        self._client = nodeclient.NodeClient(self.node_id, node_hub_host=NODE_HUB_HOST)
        self._client.register_method("RobotTask", self._on_start_call)
        _logger = log.init(APP_NAME, verbose=False)
        #init model with json file
        self.model = ACT_API(MODEL_DB,_logger)
        self.model_id = None
        self.req_id = None
        #init logger
        self.sent_image = 0
        self.total_image = 0
        self.error_image = 0

        # Start a thread to monitor the task_event
        self._monitor_thread = threading.Thread(target=self._monitor_task_event)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        #pub-sub：client2订阅client1：
        #err, topic_filters = self._client.subscribe([nodesdk.TopicFilter(topic_filter=LLM_TOPIC,
                                                                        #qos=nodesdk.MB_QOS1)])
    #push task status to nodehub
    def push(self, msg):
        rc = self._client.publish(BOT_TOPIC, payload=msg, payload_type=nodesdk.ContentType.PB,
                                  qos=nodesdk.Qos.MB_QOS1)
        if not rc:
            log.logger.error(f"Publish error")
        #else:使用文件传输助手，手机电脑轻松互传文件。


            #print("image data published to {BOT_TOPIC}")
    def send_image(self,image_data, url,time_stamp):
        
        _, img_encoded = cv2.imencode('.jpg', image_data)
        img_bytes = img_encoded.tobytes()
        compressed_img = zlib.compress(img_bytes)
        headers = {'Content-Type': 'application/octet-stream'}

        # 发送 POST 请求
        try:
            response = requests.post(url, data=compressed_img, headers=headers, timeout = 0.3)
            response.raise_for_status()
            log.logger.info(f"Send image with timestamp of {time_stamp} to {url} successfully")
            self.sent_image += 1
        except (Timeout, RequestException) as e:
            log.logger.error(f"Failed to send image with timestamp of {time_stamp} to {url} with error: {e}")
            self.error_image += 1
            return
    def _on_start_call(self, req_id, content, content_type, source, timeout):
        #log.logger.info(f"req_id: {req_id}, content: {content}, content_type: {content_type}, source: {source}, timeout: {timeout}")
        print(f"req_id: {req_id}, content: {content}, content_type: {content_type}, source: {source}, timeout: {timeout}")
        self.model_id = int(content.decode('utf-8'))# 解码为字符串，然后转换为整数
        self.req_id = req_id
        #model_id = int.from_bytes(content,'big') # content is the task ID in json
        log.logger.info(f"requied model_id: {self.model_id}")

        print(f"requested_model_id: {self.model_id}")
        
        #self._client.reply_rpc(self.req_id,b"Task Begin",nodesdk.ContentType.PB)
        #self._client.reply_rpc(self.req_id,b"Task completed",nodesdk.ContentType.PB)

        self.task_event.set()
        self._client.reply_rpc(self.req_id,b"Task Recieved",nodesdk.ContentType.PB)

    def _monitor_task_event(self):
        while True:
            self.task_event.wait()  # Wait until the event is set
            self._inference_event()  # Call the inference event
            #wait for the task to be completed
            self.task_event.clear()
    def _inference_event(self):
        if self.task_event.is_set():
            init_status = self.model.init(self.model_id)
            if not init_status:
                log.logger.error("Model not found")
                self.push(b"Model not found")
                self.task_event.clear()
                log.logger.info("Current Request Ended with Error")
                return
            self.push(b"Task Begin")
            log.logger.info("Task Begin")
            self.model.inference_start()
            #print("task started!")
            
            time_started = time.time()
            is_inference = True
            inference_time = 0
            #time.sleep(90)
            print(self.model.inference_done.is_set())
            while not self.model.inference_done.is_set() and not self.model.error_flag.is_set():
                #print(self.model.inference_done.is_set())
                
                """
                obs_data = self.model.get_obs_info()
                self.push(obs_data['images']['cam_high'].tobytes())
                print("obs_data: ",obs_data['time_stamp'])"""
                if self.model.error_flag.is_set():
                    self.push(b"Error in running inference")
                    print("Error in running inference")
                    self.task_event.clear()
                    self.model.completed_event.clear()
                    self.model.inference_done.clear()
                    self.model.process.terminate()
                    return
                else:
                    obs_data = self.model.get_obs_info()#ljy
                    log.logger.info(f"obs_data: {obs_data['time_stamp']} ")
                    image_data = obs_data['images']['cam_high']
                    self.total_image += 1
                    #print("com: %d",len(image_data))
                    self.send_image(image_data,IMAGE_URL,obs_data['time_stamp'])
                #print(len(compressed_data))
                #self.push(compressed_data)
            #print(self.model.inference_done.is_set())
            
            #self.model.inference_done.clear()
            #self.model.completed_event.clear()
            self.model.completed_event.wait()
            #print("completed_event: ",self.model.completed_event.is_set())
            
            if self.model.completed_event.is_set() and not self.model.error_flag.is_set():
                self.push(b"Task completed")    #机器人任务全部完成
                self.task_event.clear()
                self.model.completed_event.clear()
                self.model.inference_done.clear()
                self.model.process.terminate()
                time_completed = time.time()


                if self.req_id is not None:
                    #self._client.reply_rpc(self.req_id,b"Task completed",nodesdk.ContentType.PB)
                    #print("task completed!")
                    log.logger.info("Task completed")
                    self.push(b"Task completed")
                    #print(f"Time taken: {time_completed - time_started} s")
                    log.logger.info(f"Time taken: {time_completed - time_started} s")
                    log.logger.info(f"Total image captured: {self.total_image}, image sent: {self.sent_image}, image error: {self.error_image}")
                    self.total_image, self.error_image, self.sent_image = 0, 0, 0
                    print("="*40)
            elif self.model.error_flag.is_set():
                self.push(b"Error in running inference")
                self.task_event.clear()
                self.model.completed_event.clear()
                self.model.inference_done.clear()
                self.model.process.terminate()
                print("Error in running inference")
                print("="*40)

def main():
    robot_client = NodeSDKAPI()
    running = True
    while running:
            time.sleep(0.5)


if __name__ == "__main__":
    main()