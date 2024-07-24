from platform import node
import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
from ACT_API import ACT_API
import log
import time

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

    def send_rpc_request(self):
        #send RPC request to nodehub
        err, rc = self._client.send_rpc('eai.system.robot', 'StartTask', b'1', nodesdk.ContentType.PB,timeout=300)
        return err, rc                                 
    def _on_rpc_sent(_err, _req_id):
        print(f'Send RPC result: {_err}, req_id: {_req_id}')

    def _on_rpc_reply(reply):
        print(f'Receive RPC reply: {reply.req_id}, content: {reply.content}, content_type: {reply.content_type}, source: {reply.source}, timeout: {reply.timeout}')
    
def main():
    node_sdk = NodeSDKAPI()
    err, req_id = node_sdk.send_rpc_request()
    print(f'Send RPC result: {err}, req_id: {req_id}')
    while True:
        time.sleep(1)
        print("waiting for reply...")
        if req_id:
            print("req_id: ",req_id)
            break


if __name__ == "__main__":
    main()