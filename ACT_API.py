from calendar import c
from multiprocessing import Process,Event, Manager
import multiprocessing as mp
from pyexpat import model
import threading
from API_init import init_model
from API_eval import eval_bc
import time
import can
import cv2

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from log import logger
import json
from API_record import ACT_Record
import log
MODEL_DICT = "robot_task.json"
class ACT_API:
    def __init__(self, model_dict, logger=None):
        #init log
        self.config = None
        if logger is not None:
            self._logger = logger
        #model_dict is path to model json file
        #load json file as dict
        with open (model_dict,'r') as f:
            self.model_dict = json.load(f)
        #print("loaded {} models".format(len(self.model_dict)))
        if logger is not None:
            self._logger.info("loaded {} models".format(len(self.model_dict)))
        #set multiprocess method to spawn to avoid cuda error
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        #create event to signal completion of inference
        self.completed_event = Event()
        self.inference_done = Event()
        self.current_step_event = Event()
        self.current_step_end_event = Event()
        #create event to signal waiting for inference
        self.pause_event = None
        #create event to signal data ready
        self.obs_data_ready = Event()
        #create event to signal action data ready
        self.action_data_ready = Event()
        #create shared data to store inference results
        self.inference_ready=Event()#ljy
        self.error_flag = Event()
        # create event to signal waiting for start_inference
        self.info_data = Manager().dict()
        self.action_data = Manager().dict()

        
        #for record only
        self.closing_ceremony_event = None
        self.record_data = None
        self.capture_end_event = None
        self.dict_ready_event = None
        self.episode_recorder = None
        self.record_data_ready_event = None
        self.save_done = None
        self.idx = None


    #get observation data
    def get_obs_info(self):
        self.obs_data_ready.wait()
        #print("ACTAPI: obs_data_ready")
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

    def init(self, model_id):
        config_file = self.find_config_file(model_id)
        if config_file is None:
            #print("Model not found")
            return False
        else:
            self.config = init_model(config_file)
            return True

    def inference_start(self, enable_tracker = False):
        if enable_tracker:
            self.pause_event = Event()
        self.process = Process(target=self.run_inference)
        self.process.start()

    def inference_ready(self):#ljy
        self.inference_ready.set()

    def run_inference(self):
        try:
            eval_bc(config = self.config,
                    info_data = self.info_data,
                    action_data = self.action_data,
                    pause_event = self.pause_event,
                    obs_ready_event = self.obs_data_ready,
                    action_ready_event = self.action_data_ready,
                    completed_event = self.completed_event,
                    inference_done = self.inference_done,
                    current_step_event = self.current_step_event,
                    current_step_end_event = self.current_step_end_event,
                    )
            if self.completed_event.is_set():
                print("Inference completed")
                
        except Exception as error:
            #print("Error in running inference:",error)
            #print("Inference failed")
            self.error_flag.set()
            self.inference_done.set()
            self.completed_event.set()
            if self._logger is not None:
                self._logger.error("Error in running inference:",error)
            raise error
    def display_obs(self,obs_data):
        if obs_data and 'images' in obs_data:
            for camera_name, img in obs_data['images'].items():
                if img is not None:
                    cv2.imshow(camera_name, img)
    def find_config_file(self,model_id):
            for task in self.model_dict.values():
                if task['id'] == model_id:
                    return task['config_file']
            return None
    def run_record(self,):
        #init capture
        
        try:
            self.episode_recorder = ACT_Record(idx = self.idx, task_config = self.config,
                                            closing_ceremony_event = self.closing_ceremony_event,
                                            capture_end_event = self.capture_end_event,
                                            dict_ready_event = self.dict_ready_event,
                                            data_reday_event = self.record_data_ready_event,
                                            save_done_event = self.save_done,
                                            logger = self._logger,
                                            record_data = self.record_data
                                            )
            
            self.episode_recorder.capture_one_episode()
        except Exception as error:
            self._logger.error("Error in running recorder:",error)
            raise error
            

    def record_start(self):
        self.closing_ceremony_event = Event()
        self.capture_end_event = Event()
        self.dict_ready_event = Event()
        self.record_data = Manager().dict()
        self.record_data_ready_event = Event()
        self.save_done = Event()
        self.record_process = Process(target=self.run_record)
        self.record_process.start()
#API使用示例
def main():
    #实例化模型
    model = ACT_API(MODEL_DICT)
    #根据指定配置文件初始化模型
    model.init(model_id=1)

    #开始推理
    model.inference_start()
    
    query_time = 0
    #判断推理是否结束
    while not model.inference_done.is_set():
        #获取观测信息
        obs_data = model.get_obs_info()
        #显示观测信息
        #model.display_obs(obs_data)
        #获取模型动作预测
        action_data = model.get_action_info()
        # 将action_data转换为字符串并记录到日志
        #action_data_str = str(action_data)
        
        action_data['time_stamp'] = time.time()
        query_time += 1
        action_data['query_time'] = query_time
        print(action_data)
        #save action_data to txt
        with open('{}-action_data.txt'.format('transfer_straw.yaml'),'a') as f:
            f.write(str(action_data)+'\n')
        #if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下'q'键，则退出循环
        #    break
    
    



if __name__ == "__main__":

    main()
