from multiprocessing import Process,Event, Manager
import multiprocessing as mp
import threading
from API_init import init_model
from API_eval import eval_bc

class ACT_API:
    def __init__(self):
        self.config = None
        #set multiprocess method to spawn to avoid cuda error
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        self.completed_event = Event()
        self.info_data = Manager().dict()
    def init(self,config_file):
        self.config = init_model(config_file)
    def inference_start(self):
        self.process = Process(target=self.run_inference())
        self.process.start()
    def run_inference(self):
        try:
            eval_bc(self.config)
            self.completed_event.set()
        except Exception as error:
            print("Error in running inference:",error)
        

def main():
    model = ACT_API()
    model.init('transfer_straw.yaml')

    model.inference_start()
    model.completed_event.wait()
    print("Inference completed")

if __name__ == "__main__":
    main()
