
from platform import node

from networkx import within_inter_cluster
import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
from ACT_API import ACT_API
import threading
import log
import time
NODE_HUB_HOST = "127.0.0.1"
ASR_TOPIC = "/eai/system/voice"
LLM_TOPIC = "/eai/system/command"
BOT_TOPIC = "/eai/system/robot"
MODEL_DB = "robot_task.json"
class NodeSDKAPI:
    def __init__(self):
        self.task_event = threading.Event()
        self.node_id = 'eai.system.robot'
        self._client = nodeclient.NodeClient(self.node_id, node_hub_host=NODE_HUB_HOST)
        self._client.register_method("StartTask", self._on_start_call)
        self.task_status = "Waiting for task ID..."
        #init model with json file
        self.model = ACT_API(MODEL_DB)
        #pub-sub：client2订阅client1：
        #err, topic_filters = self._client.subscribe([nodesdk.TopicFilter(topic_filter=LLM_TOPIC,
                                                                        #qos=nodesdk.MB_QOS1)])
    #push task status to nodehub
    def push(self, msg):
        rc = self._client.publish(BOT_TOPIC, payload=msg, payload_type=nodesdk.ContentType.PB,
                                  qos=nodesdk.Qos.MB_QOS1)
        if not rc:
            log.logger.error(f"Publish error")
    
    def _on_start_call(self, req_id, content, content_type, source, timeout):
        #log.logger.info(f"req_id: {req_id}, content: {content}, content_type: {content_type}, source: {source}, timeout: {timeout}")
        print(f"req_id: {req_id}, content: {content}, content_type: {content_type}, source: {source}, timeout: {timeout}")
        model_id = int(content.decode('utf-8'))# 解码为字符串，然后转换为整数
        #model_id = int.from_bytes(content,'big') # content is the task ID in json
        #log.logger.info(f"model_id: {model_id}")
        print(f"requested_model_id: {model_id}")
        self.model.init(model_id)
        #self._client.reply_rpc(req_id,b"Task started",nodesdk.ContentType.PB)
        print("Task started!")
        self.model.inference_ready.set()#ljy
        #log.logger.info("Reply start")        
        print(self.model.inference_ready.is_set())
        #self.model.inference_start()
        print("task completed!")
        self.task_status = "Task completed"
        
        self._client.reply_rpc(req_id,b"Task completed",nodesdk.ContentType.PB)
    
    def start_inference(self):
        while True:
            self.model.inference_ready.wait() #ljy  # 等待事件被设置
            print("Inference started!")
            self.model.inference_start()
            
            is_inference = True
            #inference_time = 0
            #time.sleep(90)
            while not self.model.inference_done.is_set():
                print(self.model.inference_done.is_set())

                obs_data = self.model.get_obs_info()
                self.push(obs_data['images']['cam_high'].tobytes())
                print("obs_data: ",obs_data['time_stamp'])
            self.model.inference_ready.clear()
        #print(self.model.inference_done.is_set())

        

def main():
    robot_client = NodeSDKAPI()
    running = True
    while running:
            time.sleep(0.5)


if __name__ == "__main__":
    main()