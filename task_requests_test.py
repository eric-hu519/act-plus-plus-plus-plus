#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 yaqiang.sun.
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.
#########################################################################
# Author: yaqiangsun
# Created Time: 2024/07/11 10:59:21
########################################################################

from platform import node
import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
from ACT_API import ACT_API
import logging
import time
import json
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_request_test.log") 
NODE_HUB_HOST = "127.0.0.1"


class NodeSDKAPI:
    def __init__(self,data):
        self.node_id = 'eai.system.llm'
        self._client = nodeclient.NodeClient(node_id=self.node_id, node_hub_host=NODE_HUB_HOST)
        if data is not None:
            self.data_bytes = data.to_bytes(4,'big')#将int类型的data转换为4字节的bytes
            print("data is None!")
        else:
            data = 1
            print("data_bytes = b'1'")
        self.data_bytes = data.to_bytes(4,'big')#将int类型的data转换为4字节的bytes
        logger.info(f"Initialized NodeSDKAPI with llm_data: {data}")

    def send_rpc_request(self):
        #send RPC request to nodehub
        err, rc = self._client.send_rpc('eai.system.command', 'API_StartTask', self.data_bytes, nodesdk.ContentType.PB,timeout=300)
        return err, rc                                 
    def _on_rpc_sent(_err, _req_id):
        print(f'Send RPC result: {_err}, req_id: {_req_id}')

    def _on_rpc_reply(reply):
        print(f'Receive RPC reply: {reply.req_id}, content: {reply.content}, content_type: {reply.content_type}, source: {reply.source}, timeout: {reply.timeout}')
    


def main():
    # 大模型 输出端口
    url = "http://172.29.220.167:8000/command"

    payload = {"command": "先把方块换个手，再把吸管换个手，再把方块换回来"}
    headers = {"content-type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers)
    print(response.text)

    ##提取输出内容data
    response_dict = json.loads(response.text)
    data = response_dict["data"]
    print(data)

    #向eai.system.command发送RPC请求
    node_sdk = NodeSDKAPI(data)
    err, req_id = node_sdk.send_rpc_request()
    print(f'Send RPC result: {err}, req_id: {req_id}')
    while True:
        time.sleep(1)
        print("waiting for reply...")
        if req_id:
            print("req_id: ",req_id)
            time.sleep(10)
            break


if __name__ == "__main__":
    main()