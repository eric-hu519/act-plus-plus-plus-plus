from multiprocessing import Process, Event, Manager
import multiprocessing as mp
from API_init import init_model
from API_eval import eval_bc
import cv2
import numpy as np
import log
from main2 import 

class ACT_API:
    def __init__(self):
        self.config = None
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        self.completed_event = Event()
        self.inference_done = Event()
        self.pause_event = Event()
        self.obs_data_ready = Event()
        self.action_data_ready = Event()
        self.info_data = Manager().dict()
        self.action_data = Manager().dict()
        self.api = None

    def set_api(self, api):
        self.api = api

    def get_obs_info(self):
        self.obs_data_ready.wait()
        self.obs_data_ready.clear()
        return self.info_data

    def get_action_info(self):
        self.action_data_ready.wait()
        self.action_data_ready.clear()
        return self.action_data

    def wait_for_change(self):
        self.pause_event.set()

    def resueme_inference(self):
        self.pause_event.clear()

    def init(self, config_file):
        self.config = init_model(config_file)

    def inference_start(self):
        self.process = Process(target=self.run_inference)
        self.process.start()

    def run_inference(self):
        try:
            eval_bc(config=self.config,
                    info_data=self.info_data,
                    action_data=self.action_data,
                    pause_event=self.pause_event,
                    obs_ready_event=self.obs_data_ready,
                    action_ready_event=self.action_data_ready,
                    completed_event=self.completed_event,
                    inference_done=self.inference_done)
            if self.completed_event.is_set():
                print("Inference completed")
        except Exception as error:
            print("Error in running inference:", error)
            print("Inference failed")

    def display_obs(self, obs_data):
        if obs_data and 'images' in obs_data:
            for camera_name, img in obs_data['images'].items():
                if img is not None:
                    cv2.imshow(camera_name, img)
                    self.push_obs(camera_name, img)  # 推送 obs 数据

    def push_obs(self, camera_name, img):
        if self.api is not None:
            # 将图像转换为向量数据
            img_vector = self.image_to_vector(img)
            # 创建并发布消息
            msg = {'camera_name': camera_name, 'image_vector': img_vector}
            self.api.push(msg)

    @staticmethod
    def image_to_vector(img):
        # 将图像转换为向量（直接展平为一维数组）
        img_vector = img.flatten()
        return img_vector

# API使用示例
def main():
    model = ACT_API()
    model.init('transfer_straw.yaml')
    model.inference_start()
    
    while not model.inference_done.is_set():
        obs_data = model.get_obs_info()
        model.display_obs(obs_data)
        action_data = model.get_action_info()
        print(action_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()