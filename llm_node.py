#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 yaqiang.sun.
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.
#########################################################################
# Author: yaqiangsun
# Created Time: 2024/07/11 10:49:08
########################################################################


import requests

from platform import node
import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
import log
import time
import json
import requests
import threading
import queue

from sympy import N

NODE_HUB_HOST = "127.0.0.1"
ASR_TOPIC = "/eai/system/voice"
LLM_TOPIC = "/eai/system/command"
BOT_TOPIC = "/eai/system/robot"
MODEL_DB = "robot_task.json"
APP_NAME = "llm_node"

class NodeSDKAPI:
    def __init__(self):
        self.logger = log.init(APP_NAME,verbose=False)
        self.node_id = 'eai.system.llm'
        self._client = nodeclient.NodeClient(node_id=self.node_id, node_hub_host=NODE_HUB_HOST)
        self._client.set_on_message(self._on_message)#ljy
        #subscrib voice node
        self.subscrib_voice(ASR_TOPIC)
        
        #llm response event
        self.response_event = threading.Event()
        # Start a thread to monitor the task_event
        self._monitor_thread = threading.Thread(target=self._monitor_task_event)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()



    def subscrib_voice(self,node_name: str):
        err,topic_filter = self._client.subscribe(
            [nodesdk.TopicFilter(topic_filter=node_name, qos=nodesdk.Qos.MB_QOS1)]
        )#ljy
        if err!= nodesdk.ClientErr.OK:
            log.logger.error(f"Subscribe error: {err}")
            #print("Subscribe error: {err}")
            return#ljy
        log.logger.info(f"Subscribe Success")

    def push(self, msg):
        rc = self._client.publish(LLM_TOPIC, payload=msg, payload_type=nodesdk.ContentType.PB,
                                  qos=nodesdk.Qos.MB_QOS1)
        if not rc:
            log.logger.error(f"Publish error")

    def get_response_from_llm(self):
        
        url = "http://172.29.220.167:4397/api/stream"
        response = requests.request("GET", url,stream=True)
        # print(response.status_code)
        if "data:" in response:
            response = response.split("data:")[1].strip()
        log.logger.info(f"Receive task name from llm: {response}")
        return response
        
    def send_request(self, payload: str):
        #self.event.wait()
        url = "http://172.29.220.167:8000/command"
        payload = {"command": payload}
        headers = {"content-type": "application/json"}
        requests.request("POST", url, json=payload, headers=headers)
        return
    
    def _monitor_task_event(self):
        while True:
            self.response_event.wait()  # Wait until the event is set
            reponse = self.get_response_from_llm()
            model_id = self.parse_requst(reponse, MODEL_DB)
            if model_id is not None:
                err, rc = self.send_rpc_request(model_id)
                log.logger.info(f'Send RPC result: {err}, req_id: {rc}')
            #wait for the task to be completed
            self.response_event.clear()

    def parse_requst(self, task_name: str, model_db: str):
        #find required model_id in MODEL_DB
        with open(model_db, 'r') as f:
            model_db = json.load(f)
        if task_name in model_db:
            model_id = model_db[task_name].get('id')
            return model_id
        else:
            log.logger.error(f"Task name \"{task_name}\" not found error")
            return

    def _on_message(self, msg):
        #recieve voice message and send
        print(f"Receive message on topic {msg.topic}")
        self.response_event.set()
        payload = msg.payload.decode('utf-8')
        print(f"Receive voice message: {payload}")
        log.logger.info(f"Receive voice message: {payload}")
        #test only
        payload = "先把方块换个手，再把吸管换个手，再把方块换回来"
        self.send_request(payload)


    def send_rpc_request(self, model_id: int):
        #send RPC request to nodehub
        model_id_byte = str(model_id).encode('utf-8')
        err, rc = self._client.send_rpc('eai.system.robot', 'RobotTask', model_id_byte, nodesdk.ContentType.PB,timeout=300)
        return err, rc                                 
    def _on_rpc_sent(_err, _req_id):
        print(f'Send RPC result: {_err}, req_id: {_req_id}')

    def _on_rpc_reply(reply):
        print(f'Receive RPC reply: {reply.req_id}, content: {reply.content}, content_type: {reply.content_type}, source: {reply.source}, timeout: {reply.timeout}')
    

def main():
    # 大模型 输出端口
    llm_client = NodeSDKAPI()
    running = True
    while running:
        time.sleep(0.5)


if __name__ == "__main__":
    main()