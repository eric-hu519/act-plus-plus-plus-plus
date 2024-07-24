import nodeclient.nodeclient as nodeclient
import nodesdk_py as nodesdk
from ACT_API import ACT_API
import log

NODE_HUB_HOST = "127.0.0.1"
ASR_TOPIC = "/eai/system/voice"
LLM_TOPIC = "/eai/system/command"
BOT_TOPIC = "/eai/system/robot"

class NodeSDKAPI:
    def __init__(self, node_id):
        self._client = nodeclient.NodeClient(node_id, node_hub_host=NODE_HUB_HOST)
        self._client.register_method("StartTask", self._on_start_call)
        self.task_status = "Waiting for task ID..."

        self.model = ACT_API()

        err, topic_filters = self._client.subscribe([nodesdk.TopicFilter(topic_filter=LLM_TOPIC,
                                                                        qos=nodesdk.MB_QOS1)])
    #push task status to nodehub
    def push(self, msg):
        rc = self._client.publish(BOT_TOPIC, payload=msg, payload_type=nodesdk.ContentType.PB,
                                  qos=nodesdk.Qos.MB_QOS1)
        if not rc:
            log.logger.error(f"Publish error")
    
    def _on_start_call(self, req_id, content, content_type, source, timeout):
        log.logger.info(f"req_id: {req_id}, content: {content}, content_type: {content_type}, source: {source}, timeout: {timeout}")
        model_id = content
        self.model.init(content)
        self.model.inference_start()
        self.task_status = "Task started"
        while self.model.inference_done.is_set():
            obs_data = self.model.get_obs_info()
            self.model.display_obs(obs_data)
            action_data = self.model.get_action_info()
            self.task_status = action_data
            self.push(action_data)
        self.task_status = "Task completed"
        self.client.reply_rpc("Task completed")
        