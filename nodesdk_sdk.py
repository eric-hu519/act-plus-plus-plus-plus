from enum import Enum

import nodesdk_py as nodesdk

import log
import voice_pb2

NODE_HUB_HOST = "127.0.0.1"
ASR_TOPIC = "/eai/system/voice"


class Event(Enum):
    START = 0
    STOP = 1


class NodeSDKAPI:
    def __init__(self, node_id, cmd_timeout_ms=500, wait_message_time_ms=500, save_logs=False):
        """
        创建NodeSDK客户端
        :param node_id: 节点ID
        :param cmd_timeout_ms: 命令超时时间
        :param wait_message_time_ms: 数据接收时间（命令发送间隔时间）
        :param save_logs: 是否保存日志到文件
        """
        self._on_event = None
        self._client = nodesdk.NodeSDKClient()
        log.logger.debug(f"NodeSDK Client Version, {self._client.GetSoftVersion()}")
        self._client.Init(cmd_timeout_ms, wait_message_time_ms, save_logs)
        self._node_id = node_id
        self._client.SetOnRpcMethod(self._on_method_call)
        self.is_ready = False

    def __del__(self):
        self._client.DeInit()

    def deinit(self):
        self._client.Disconnect()
        self.is_ready = False

    def init(self):
        log.logger.info("Connecting to NodeHub")
        return self._connect(NODE_HUB_HOST, options=None)

    def _on_connect(self, err, ack):
        log.logger.info(f"Connect err: {err}, ack code: {ack.code}")
        if ((err == nodesdk.ClientErr.OK or err == nodesdk.ClientErr.CONNECTED)
                and ack.code == nodesdk.ReasonCode.SUCCESS):
            log.logger.info("Connect NodeHub success")
            self.is_ready = True
        else:
            # 连接失败5s钟后重试
            self._reconnect(5)

    def _on_disconnect(self, code):
        log.logger.error(f"Disconnect code: {code}")
        self.is_ready = False
        self._reconnect(5)

    def _reconnect(self, second):
        log.logger.info(f"Reconnecting to NodeHub after {second}s")
        return self._client.Reconnect(second * 1000, self._on_connect)

    def _connect(self, host, port=1883, options=None):
        """
        连接到NodeHub
        :param host: NodeHub地址
        :param port: NodeHub端口，默认1883
        :param options: 可选参数
            user_name: 用户名
            password: 密码
            keep_alive: 保持心跳时间，单位：s
            rpc: 是否启用RPC，默认启用
            broadcast: 是否启用广播，默认禁用
            clean: 连接时是否清除回话，默认清除
            timeout_ms: 连接超时时间，单位：ms，默认60000
        :return:
        """
        if options is None:
            options = {}
        connect_info = nodesdk.ConnectInfo()
        connect_info.node_id = self._node_id
        if options.get("user_name") is not None:
            connect_info.user_name = options.get("user_name")
        if options.get("password") is not None:
            connect_info.password = options.get("password")
        if options.get("keep_alive") is not None:
            connect_info.keep_alive = options.get("keep_alive")
        else:
            connect_info.keep_alive = 60
        if options.get("rpc") is not None:
            connect_info.rpc = options.get("rpc")
        else:
            connect_info.rpc = True
        if options.get("broadcast") is not None:
            connect_info.broadcast = options.get("broadcast")
        else:
            connect_info.broadcast = False
        if options.get("clean") is not None:
            connect_info.clean = options.get("clean")
        else:
            connect_info.clean = True

        endpoint = nodesdk.Endpoint()
        endpoint.host = host
        endpoint.port = port
        if options.get("timeout_ms") is not None:
            endpoint.timeout_ms = options.get("timeout_ms")
        else:
            endpoint.timeout_ms = 5000

        rc = self._client.Connect(endpoint, connect_info, self._on_connect, self._on_disconnect)
        if rc != nodesdk.ClientErr.OK:
            log.logger.error(f"Connect error: {rc}")
            return False
        return True

    def push(self, raw, text):
        asr_result = voice_pb2.AsrResult()
        asr_result.raw = raw
        asr_result.text = text
        message = nodesdk.Message()
        message.payload_type = nodesdk.ContentType.PB
        message.payload = asr_result.SerializeToString()
        message.topic = ASR_TOPIC
        message.qos = nodesdk.Qos.MB_QOS1
        message.retain = False

        rc = self._client.Publish(message)
        if rc != nodesdk.ClientErr.OK:
            log.logger.error(f"Publish error: {rc}")

    def set_on_event(self, on_event):
        """
        订阅消息回调
        :param on_event:
            def on_event(event, msg)
        :return:
        """
        self._on_event = on_event

    def _on_method_call(self, method):
        log.logger.info(f"Method: {method.method}, Source: {method.source}, ContentType: {method.content_type}")
        reply = nodesdk.RpcReply()
        reply.req_id = method.req_id
        reply.content_type = nodesdk.ContentType.PB
        if method.content_type != nodesdk.ContentType.PB:
            log.logger.warn("ContentType is not PB")
            reply.code = nodesdk.ReasonCode.CONTENT_TYPE_NOT_SUPPORT
        else:
            try:
                if method.method == "start":
                    if self._on_event is not None:
                        self._on_event(Event.START, None)
                elif method.method == "stop":
                    if self._on_event is not None:
                        self._on_event(Event.STOP, None)
                else:
                    reply.code = nodesdk.ReasonCode.METHOD_NOT_FOUND
            except Exception as e:
                log.logger.error(f"Method call error: {e}")
                reply.code = nodesdk.ReasonCode.UNDEFINED

        self._client.ReplyRpc(reply)
        log.logger.info("Reply Rpc")
