from platform import node
import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
from ACT_API import ACT_API
import logging
import time
import cv2
import numpy as np
import gzip #ljy
import sys
import log

logging.basicConfig(level=logging.INFO) #ljy
logger = logging.getLogger("robot.main")#ljy

NODE_HUB_HOST = "127.0.0.1"
ASR_TOPIC = "/eai/system/voice"
LLM_TOPIC = "/eai/system/command"
BOT_TOPIC = "/eai/system/robot"
MODEL_DB = "/home/mamager/Documents/casia-bot-main-nodes-voice/nodes/voice/tests/robot_task.json"
APP_NAME = "llm_node"
TRACKER_TOPIC = "/eai/system/visualtrack/position"
TRACKER_ID = "eai.system.track"
BOT_ID = "eai.system.robot"
#test file for sending LLM results and receiving RPC callbacks
class NodeSDKAPI:
    def __init__(self):
        self.node_id = 'eai.system.command'
        self._client = nodeclient.NodeClient(node_id=self.node_id, node_hub_host=NODE_HUB_HOST)

        err,topic_filter = self._client.subscribe(
            [nodesdk.TopicFilter(topic_filter=BOT_TOPIC, qos=nodesdk.Qos.MB_QOS1)],
            self._on_subscribe
        )#ljy
        err,topic_filter = self._client.subscribe(
            [nodesdk.TopicFilter(topic_filter=ASR_TOPIC, qos=nodesdk.Qos.MB_QOS1)],
            self._on_subscribe)

        if err!= nodesdk.ClientErr.OK:
            logger.error(f"Subscribe error: {err}")
            print("Subscribe error: {err}")
            return#ljy
        logging.info(f"Subscribe Success")
       
        self._client.set_on_message(self._on_message)#ljy
        self.cnt = 0

    def send_rpc_request(self,content,node_id:str, method:str) -> tuple:
        #send RPC request to nodehub
        if isinstance(content, int):
            send_content = str(content).encode('utf-8')
        else:
            send_content = content.encode('utf-8')
        err, rc = self._client.send_rpc(node_id, method, send_content, nodesdk.ContentType.PB,timeout=300)
        return err, rc   


    # def send_rpc_request(self):
    #     #send RPC request to nodehub
    #     err, rc = self._client.send_rpc('eai.system.robot', 'RobotTask', b'1', nodesdk.ContentType.PB,timeout=300)
    #     return err, rc                                 
    def _on_rpc_sent(_err, _req_id):
        print(f'Send RPC result: {_err}, req_id: {_req_id}')

    def _on_rpc_reply(reply):
        print(f'Receive RPC reply: {reply.req_id}, content: {reply.content}, content_type: {reply.content_type}, source: {reply.source}, timeout: {reply.timeout}')
    
    def _on_message(self, msg):#ljy
        print(f"Receive message on topic {msg.topic}")
        #self.cnt +=1
        #print(self.cnt)

    
    @staticmethod
    def _on_subscribe(err,topic_filters):
        if err == nodesdk.ClientErr.OK:
            logging.info(f"Subscribed to topics: {topic_filters}")
        else:
            logging.error(f"Subscribed failed with error: {err}")


def main(test_record = True):
    node_sdk = NodeSDKAPI()
    print(f"Sending rpc to robot node and  tracker node")
    model_id = "1"
    record_id = "1,-1"
    object_name = "red_straw"
    if not test_record:
        err, rc = node_sdk.send_rpc_request(content=model_id, node_id=BOT_ID, method="RobotTask")
    else:
        err,rc = node_sdk.send_rpc_request(content=record_id, node_id=BOT_ID, method="RecordTask")
    #if not self.test:
    print(f'Send ROBOT RPC result: {err}, req_id: {rc}')
    _err, _rc = node_sdk.send_rpc_request(content=object_name, node_id=TRACKER_ID, method="VisualTrack")
    if err != nodesdk.ClientErr.OK:
        print(f"Send RPC to robot node failed: {err}")
    else:
        print(f"Send RPC to robot node successfully")
        
    if _err != nodesdk.ClientErr.OK:#ljy
        print(f"Send RPC to tracker node failed: {err}")
    else:
        print(f"Send RPC to tracker node successfully")
        
    
    while True:
        pass
        """time.sleep(1)
        print("waiting for reply...")
        if req_id:
            print("req_id: ",req_id)
            break"""


if __name__ == "__main__":
    main()