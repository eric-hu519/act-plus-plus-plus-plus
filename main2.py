from platform import node
import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
from ACT_API import ACT_API
import logging
import time
import cv2
import numpy as np
import gzip #ljy

logging.basicConfig(level=logging.INFO) #ljy
logger = logging.getLogger("robot.main")#ljy

NODE_HUB_HOST = "127.0.0.1"
ASR_TOPIC = "/eai/system/voice"
LLM_TOPIC = "/eai/system/command"
BOT_TOPIC = "/eai/system/robot"
MODEL_DB = "robot_task.json"
#test file for sending LLM results and receiving RPC callbacks
class NodeSDKAPI:
    def __init__(self):
        self.node_id = 'eai.system.command'
        self._client = nodeclient.NodeClient(node_id=self.node_id, node_hub_host=NODE_HUB_HOST)

        err,topic_filter = self._client.subscribe(
            [nodesdk.TopicFilter(topic_filter=BOT_TOPIC, qos=nodesdk.Qos.MB_QOS1)],
            self._on_subscribe
        )#ljy
        print(err)
        if err!= nodesdk.ClientErr.OK:
            logger.error(f"Subscribe error: {err}")
            print("Subscribe error: {err}")
            return#ljy
        else:
            logging.info(f"Subscribe Success")
       
        self._client.set_on_message(self._on_message)#ljy
        self.cnt = 0


    def send_rpc_request(self):
        #send RPC request to nodehub
        err, rc = self._client.send_rpc('eai.system.robot', 'RobotTask', b'1', nodesdk.ContentType.PB,timeout=300)
        return err, rc                                 
    def _on_rpc_sent(_err, _req_id):
        print(f'Send RPC result: {_err}, req_id: {_req_id}')

    def _on_rpc_reply(reply):
        print(f'Receive RPC reply: {reply.req_id}, content: {reply.content}, content_type: {reply.content_type}, source: {reply.source}, timeout: {reply.timeout}')
    
    def _on_message(self, msg):#ljy
        logger.info(f"Receive message on topic {msg.topic}")
        #print(f"Receive message on topic {msg.topic}")
        #log message
        #print(f"Message: {msg.payload.decode('utf-8')}")
        logger.info(f"Message: {msg.payload.decode('utf-8')}")
        #self.cnt +=1
        #print(self.cnt)
        '''
        try:
            #print(len(msg.payload))
            image_data = gzip.decompress(msg.payload)
            #print("compressed:%d",len(image_data))
            image_data = np.frombuffer(image_data, dtype=np.uint8)
            #print(image_data.shape)
            image = image_data.reshape((480,640,3))
            #image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is not None:
                cv2.imshow("Robot Camera Feed", image)
                cv2.waitKey(1)
            else:
                print("Failed to decode image")
                logger.error("Failed to decode image")
        except Exception as e:
            logger.error(f"Error processing image message: {e}")
        '''
    
    @staticmethod
    def _on_subscribe(err,topic_filters):
        if err == nodesdk.ClientErr.OK:
            logging.info(f"Subscribed to topics: {topic_filters}")
        else:
            logging.error(f"Subscribed failed with error: {err}")


def main():
    node_sdk = NodeSDKAPI()
    err, req_id = node_sdk.send_rpc_request()
    print(f'Send RPC result: {err}, req_id: {req_id}')
    while True:
        pass
        """time.sleep(1)
        print("waiting for reply...")
        if req_id:
            print("req_id: ",req_id)
            break"""


if __name__ == "__main__":
    main()