from multiprocessing import Process, Event, Manager
import multiprocessing as mp
import signal
import threading
from API_init import init_model
from API_eval import eval_bc
import time
import cv2
import log
from nodesdk_api import NodeSDKAPI, Event as NodeEvent

class ACT_API:
    def __init__(self, config_file, node_id):
        self.config = None
        self.config_file = config_file
        self.node_id = node_id
        self.api = NodeSDKAPI(node_id=node_id, save_logs=False)
        
        # set multiprocess method to spawn to avoid cuda error
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        
        # create event to signal completion of inference
        self.completed_event = Event()
        self.inference_done = Event()
        # create event to signal waiting for inference
        self.pause_event = Event()
        # create event to signal data ready
        self.obs_data_ready = Event()
        # create event to signal action data ready
        self.action_data_ready = Event()
        # create shared data to store inference results
        self.info_data = Manager().dict()
        self.action_data = Manager().dict()
        
        # initialize NodeSDK
        self.init_nodesdk()
        
        # signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        self.running = False
        self.start_event = threading.Event()
        
    def init_nodesdk(self):
        log.logger.info("Connecting to NodeHub")
        self.api.init()
        self.api.set_on_event(self.on_event)
    
    def signal_handler(self, signal, frame):
        log.logger.info('CTL+C pressed')
        self.running = False
    
    def on_event(self, event, data):
        log.logger.debug(f'Robot Arm Node event: {event}')
        if event == NodeEvent.START:
            self.resume_inference()
        elif event == NodeEvent.STOP:
            self.pause_event.set()
    
    def init_model(self):
        self.config = init_model(self.config_file)

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

    # get observation data
    def get_obs_info(self):
        self.obs_data_ready.wait()
        self.obs_data_ready.clear()
        return self.info_data

    # get action data
    def get_action_info(self):
        self.action_data_ready.wait()
        self.action_data_ready.clear()
        return self.action_data

    # wait for obs change
    def wait_for_change(self):
        self.pause_event.set()

    # resume inference
    def resume_inference(self):
        self.pause_event.clear()

    def display_obs(self, obs_data):
        if obs_data and 'images' in obs_data:
            for camera_name, img in obs_data['images'].items():
                if img is not None:
                    cv2.imshow(camera_name, img)
    
    def run(self):
        log.logger.debug('Robot arm node start')
        self.running = True

        self.init_model()
        self.inference_start()

        while self.running:
            if self.start_event.is_set():
                log.logger.debug('Start robot arm in main thread')
                self.resume_inference()
                self.start_event.clear()
            else:
                # Display observation data
                obs_data = self.get_obs_info()
                self.display_obs(obs_data)
                # Get action data
                action_data = self.get_action_info()
                print(action_data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

            time.sleep(0.5)

        self.api.deinit()
        log.logger.debug('Robot arm node exit!')

if __name__ == "__main__":
    import argparse
    
    def parser_args():
        parser = argparse.ArgumentParser(description='Robot Arm Control Node')
        parser.add_argument('-c', '--config', type=str, help='Specify the configuration file for the robot arm', required=True)
        parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Print detailed logs')
        parser.add_argument('-d', '--debug', default=False, action='store_true', help='Print debug logs')
        parser.add_argument('-n', '--node_id', type=str, help='Specify the Node ID for NodeSDK', required=True)
        return parser.parse_args()
    
    args = parser_args()
    log.init('robot-arm-node', verbose=args.verbose | args.debug)
    
    model = ACT_API(config_file=args.config, node_id=args.node_id)
    model.run()