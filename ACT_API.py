from multiprocessing import Process,Event, Manager
import multiprocessing as mp
import threading
from API_init import init_model
from API_eval import eval_bc
import time
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
class ACT_API:
    def __init__(self):
        self.config = None
        #set multiprocess method to spawn to avoid cuda error
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        #create event to signal completion of inference
        self.completed_event = Event()
        self.inference_done = Event()
        #create event to signal waiting for inference
        self.pause_event = Event()
        #create event to signal data ready
        self.obs_data_ready = Event()
        #create event to signal action data ready
        self.action_data_ready = Event()
        #create shared data to store inference results
        self.info_data = Manager().dict()
        self.action_data = Manager().dict()

    #get observation data
    def get_obs_info(self):
        self.obs_data_ready.wait()
        self.obs_data_ready.clear()
        return self.info_data
    #get action data
    def get_action_info(self):
        self.action_data_ready.wait()
        self.action_data_ready.clear()
        return self.action_data
    #wait for obs change
    def wait_for_change(self):
        self.pause_event.set()

    #resume inference
    def resueme_inference(self):
        self.pause_event.clear()

    def init(self,config_file):
        self.config = init_model(config_file)

    def inference_start(self):
        self.process = Process(target=self.run_inference)
        self.process.start()

    def run_inference(self):
        try:
            eval_bc(config = self.config,
                    info_data = self.info_data,
                    action_data = self.action_data,
                    pause_event = self.pause_event,
                    obs_ready_event = self.obs_data_ready,
                    action_ready_event = self.action_data_ready,
                    completed_event = self.completed_event,
                    inference_done = self.inference_done)
            if self.completed_event.is_set():
                print("Inference completed")
        except Exception as error:
            print("Error in running inference:",error)
            print("Inference failed")
    def display_obs(self,obs_data):
        if obs_data and 'images' in obs_data:
            for camera_name, img in obs_data['images'].items():
                if img is not None:
                    cv2.imshow(camera_name, img)

#API使用示例
def main():
    #实例化模型
    model = ACT_API()
    #根据指定配置文件初始化模型
    model.init('transfer_straw.yaml')

    #开始推理
    model.inference_start()
    
    #判断推理是否结束
    while not model.inference_done.is_set():
        #获取观测信息
        obs_data = model.get_obs_info()
        #显示观测信息
        model.display_obs(obs_data)
        #获取模型动作预测
        action_data = model.get_action_info()

        print(action_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下'q'键，则退出循环
            break
        
    



if __name__ == "__main__":
    main()
