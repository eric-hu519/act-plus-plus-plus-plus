#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 yaqiang.sun.
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.
#########################################################################
# Author: yaqiangsun
# Created Time: 2024/07/11 10:59:21
########################################################################


import requests

url = "http://172.29.220.167:8000/command"

payload = {"command": "先把方块换个手，再把吸管换个手，再把方块换回来"}
headers = {"content-type": "application/json"}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)