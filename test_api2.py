
from platform import node
import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
from ACT_API import ACT_API
import threading
import log
from enum import Enum
NODE_HUB_HOST = "127.0.0.1"
ASR_TOPIC = "/eai/system/voice"
LLM_TOPIC = "/eai/system/command"
BOT_TOPIC = "/eai/system/robot"
MODEL_DB = "robot_task.json"
class Event(Enum):
    START = 0
    STOP = 1

class NodeSDKAPI:
    def __init__(self):
        self.node_id = 'eai.system.robot'
        self._on_event = None
        self._client = nodeclient.NodeClient(self.node_id, node_hub_host=NODE_HUB_HOST)
        self._client.register_method("StartTask", self._on_start_call)
        #init model with json file
        self.model = ACT_API(MODEL_DB)
        self.model_id = None
        #pub-sub：client2订阅client1：
        #err, topic_filters = self._client.subscribe([nodesdk.TopicFilter(topic_filter=LLM_TOPIC,
                                                                        #qos=nodesdk.MB_QOS1)])
    #push task status to nodehub
    def push(self, msg):
        rc = self._client.publish(BOT_TOPIC, payload=msg, payload_type=nodesdk.ContentType.PB,
                                  qos=nodesdk.Qos.MB_QOS1)
        if not rc:
            log.logger.error(f"Publish error")
    def set_on_event(self, on_event):
        """
        订阅消息回调
        :param on_event:
            def on_event(event, msg)
        :return:
        """
        self._on_event = on_event
    def _on_start_call(self, req_id, content, content_type, source, timeout):
        #log.logger.info(f"req_id: {req_id}, content: {content}, content_type: {content_type}, source: {source}, timeout: {timeout}")
        print(f"req_id: {req_id}, content: {content}, content_type: {content_type}, source: {source}, timeout: {timeout}")
        self.model_id = int(content.decode('utf-8'))# 解码为字符串，然后转换为整数
        #model_id = int.from_bytes(content,'big') # content is the task ID in json
        #log.logger.info(f"model_id: {model_id}")
        print(f"requested_model_id: {self.model_id}")
        if self._on_event is not None:
            self._on_event(Event.START, None)
        #log.logger.info("Reply start")        
        return req_id
    def start(self):
        if self.model_id is not None:
            self.model.init(self.model_id)
            self.model.inference_start()
            inference_time = 0
            while not self.model.inference_done.is_set():##这里的push怎么用？
                obs_data = self.model.get_obs_info()
                if inference_time == 0 and obs_data is not None:
                    obs_data_buffer = obs_data
                    inference_time += 1
                    self.push(obs_data_buffer['images']['cam_high'].tobytes())
                elif obs_data_buffer != obs_data:
                    obs_data_buffer = obs_data
                    self.push(obs_data_buffer['images']['cam_high'].tobytes())
            self.task_status = "Task completed"
            
            print("Task completed")
    def _on_task_end_call(self,req_id):
            self._client.reply_rpc(req_id,b"Task completed",nodesdk.ContentType.PB)
            if self._on_event is not None:
                self._on_event(Event.STOP, None)

class Main:
    def __init__(self):
        self.running = False
        self.start_event = threading.Event()
        self.robot_client = NodeSDKAPI()
    def on_event(self,event):
        if event == Event.START:
            self.robot_client.start()
            self.start_event.set()
        elif event == Event.STOP:
            self.running = False
            self.robot_client._on_task_end_call()
    def run(self):
        self.robot_client.set_on_event(self.on_event)
        self.robot_client.start()
if __name__ == "__main__":
    main()