import os
from re import T
import time
import h5py
import argparse
import h5py_cache
import numpy as np
from tqdm import tqdm
import cv2
from constants import DT, START_ARM_POSE, FPS
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from aloha_scripts.robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from aloha_scripts.robot_utils import move_arms, torque_on, torque_off, move_grippers
from aloha_scripts.real_env import make_real_env, get_action
import log
import rospy
from threading import Event
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from reset_api import closing_ceremony

class ACT_Record:
    def __init__(self, task_config, idx=None,
                 closing_ceremony_event=None,
                 capture_end_event=None,
                 dict_ready_event=None,
                 data_reday_event=None,
                 save_done_event=None,
                 record_data=None,
                 logger=None):
        
        self.data_dict = record_data
        self.data_reday_event = data_reday_event
        self.save_done_event = save_done_event
        self.dataset_dir = task_config['dataset_dir']
        self.max_timesteps = task_config['episode_len']
        self.dataset_path = None
        self.padded_size = None
        self.compressed_len = None
        self.camera_names = task_config['camera_names']
        self.closing_ceremony_event = closing_ceremony_event
        self.capture_end_event = capture_end_event
        self.dict_ready_event = dict_ready_event
        self.logger = logger
        self.gripper_close_event = Event()
        if idx is None:
            try:
                self.idx = self.get_auto_index(self.dataset_dir)
            except Exception as e:
                self.logger.error(e)
                exit()
        else:
            logger.info(f'Using idx: {idx}')
            self.idx = idx

        self.overwrite = True
        self.compress = True

        
        self.dataset_name = f'episode_{self.idx}'
        self.logger.info(f'Dataset name: {self.dataset_name}')

            


        
    
    def get_auto_index(self,dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
        max_idx = 1000
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
        for i in range(max_idx+1):
            if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
                return i
        raise Exception(f"Error getting auto index, or more than {max_idx} episodes")
    def capture_one_episode(self):
        self.master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                       robot_name=f'master_left', init_node=True)
        self.logger.info(f'Initialized master_left')
        self.master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                        robot_name=f'master_right', init_node=False)
        self.logger.info(f'Initialized master_right')
        self.env = make_real_env(init_node=False, setup_robots=False,is_record=True)
        self.logger.info(f'Initialized env')
        if not os.path.isdir(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.dataset_path = os.path.join(self.dataset_dir, self.dataset_name)
        if os.path.isfile(self.dataset_path) and not self.overwrite:
            self.logger.info(f'Dataset already exist at \n{self.dataset_path}\nHint: set overwrite to True.')
            raise Exception(f'Dataset already exist at \n{self.dataset_path}\nHint: set overwrite to True.')
        
        # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
        self.opening_ceremony(self.master_bot_left, self.master_bot_right, self.env.puppet_bot_left, self.env.puppet_bot_right)

        # Data collection
        ts = self.env.reset(fake=True)
        timesteps = [ts]
        actions = []
        actual_dt_history = []
        time0 = time.time()
        DT = 1 / FPS
        self.logger.info(f'Begin data collection')
        for t in tqdm(range(self.max_timesteps)):
            t0 = time.time()
            action = get_action(self.master_bot_left, self.master_bot_right)
            t1 = time.time()
            ts = self.env.step(action)
            t2 = time.time()
            timesteps.append(ts)
            actions.append(action)
            actual_dt_history.append([t0, t1, t2])
            time.sleep(max(0, DT - (time.time() - t0)))
        self.logger.info(f'Avg fps: {self.max_timesteps / (time.time() - time0)}')

        # Torque on both master bots
        torque_on(self.master_bot_left)
        torque_on(self.master_bot_right)
        # Open puppet grippers
        self.env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
        self.env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
        move_grippers([self.env.puppet_bot_left, self.env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

        freq_mean = self.print_dt_diagnosis(actual_dt_history)
        if freq_mean < 30:
            self.logger.info(f'\n\nfreq_mean is {freq_mean}, lower than 30, re-collecting... \n\n\n\n')
            raise Exception(f'freq_mean is {freq_mean}, lower than 30, re-collecting...')

            """
        For each timestep:
        observations
        - images
            - cam_high          (480, 640, 3) 'uint8'
            - cam_low           (480, 640, 3) 'uint8'
            - cam_left_wrist    (480, 640, 3) 'uint8'
            - cam_right_wrist   (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'
        
        action                  (14,)         'float64'
        base_action             (2,)          'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/effort': [],
            '/action': [],
            '/base_action': [],
            # '/base_action_t265': [],
        }
        for cam_name in self.camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []
        while actions:
            action = actions.pop(0)
            ts = timesteps.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/observations/effort'].append(ts.observation['effort'])
            data_dict['/action'].append(action)
            data_dict['/base_action'].append(ts.observation['base_vel'])
            # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
            for cam_name in self.camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
        if self.compress:
            # JPEG compression
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            compressed_len = []
            for cam_name in self.camera_names:
                image_list = data_dict[f'/observations/images/{cam_name}']
                compressed_list = []
                compressed_len.append([])
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image, encode_param)
                    compressed_list.append(encoded_image)
                    compressed_len[-1].append(len(encoded_image))
                data_dict[f'/observations/images/{cam_name}'] = compressed_list
            self.logger.info(f'compression: {time.time() - t0:.2f}s')
            #pad so it has same length
            t0 = time.time()
            compressed_len = np.array(compressed_len)
            self.padded_size = compressed_len.max()
            for cam_name in self.camera_names:
                compressed_image_list = data_dict[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(self.padded_size, dtype='uint8')
                    image_len = len(compressed_image)
                    padded_compressed_image[:image_len] = compressed_image
                    padded_compressed_image_list.append(padded_compressed_image)
                data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
            self.logger.info(f'padding: {time.time() - t0:.2f}s')
        self.compressed_len = compressed_len
        self.data_dict = data_dict
        self.dict_ready_event.set()

        self.data_reday_event.wait()
        self.save_h5py(self.data_dict)

        self.closing_ceremony_event.wait()
        self.record_closing_ceremony()
        self.logger.info(f'Finished recording')
        self.capture_end_event.set()
        
    def save_h5py(self, data_dict):
    # dataset_name = f'episode_{get_auto_index(dataset_dir)}'
        t0 = time.time()
        if self.dataset_path is None:
            self.logger.error('dataset_path is None')
            raise Exception('dataset_path is None')
        if self.padded_size is None:
            self.logger.error('padded_size is None')
            raise Exception('padded_size is None')
        with h5py.File(self.dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = False
            root.attrs['compress'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in self.camera_names:
                _ = image.create_dataset(cam_name, (self.max_timesteps, self.padded_size), dtype='uint8',
                                            chunks=(1, self.padded_size), )
            _ = obs.create_dataset('qpos', (self.max_timesteps, 14))
            _ = obs.create_dataset('qvel', (self.max_timesteps, 14))
            _ = obs.create_dataset('effort', (self.max_timesteps, 14))
            _ = root.create_dataset('action', (self.max_timesteps, 14))
            _ = root.create_dataset('base_action', (self.max_timesteps, 2))
            # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

            for name, array in data_dict.items():
                root[name][...] = array

            _ = root.create_dataset('compress_len', (len(self.camera_names), self.max_timesteps))
            root['/compress_len'][...] = self.compressed_len
            self.save_done_event.set()
    def opening_ceremony(self,master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right,episode_end=False):
        """ Move all 4 robots to a pose where it is easy to start demonstration """
        # reboot gripper motors, and set operating modes for all motors
        puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
        puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
        puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
        master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
        # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

        puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
        puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
        puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
        master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
        # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

        torque_on(puppet_bot_left)
        torque_on(master_bot_left)
        torque_on(puppet_bot_right)
        torque_on(master_bot_right)

        # move arms to starting position
        start_arm_qpos = START_ARM_POSE[:6]
        move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=1.5)
        # move grippers to starting position
        move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)


        # press gripper to start data collection
        # disable torque for only gripper joint of master robot to allow user movement
        master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
        master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
        print(f'Close the gripper to start')
        close_thresh = -0.1
        if not episode_end:
            while not self.gripper_close_event.is_set():
                gripper_pos_left = get_arm_gripper_positions(master_bot_left)
                gripper_pos_right = get_arm_gripper_positions(master_bot_right)
                if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
                    self.gripper_close_event.set()
                time.sleep(DT/10)
            torque_off(master_bot_left)
            torque_off(master_bot_right)

        # right_gripper_info = puppet_bot_right.dxl.srv_get_info("single", "gripper")
        # left_gripper_info = puppet_bot_left.dxl.srv_get_info("single", "gripper")
        # print('\npuppet bot griper mode, left:{}, right:{}\n'.format(left_gripper_info.mode, right_gripper_info.mode))

        self.logger.info(f'Started!')

    def reset_recording(self):
        self.dataset_path = None
        self.padded_size = None
        self.capture_end_event.clear()
        self.dict_ready_event.clear()
        self.data_reday_event.clear()
        self.save_done_event.clear()
        self.data_dict = None
        
        self.compressed_len = None
    def print_dt_diagnosis(self,actual_dt_history):
        actual_dt_history = np.array(actual_dt_history)
        get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
        step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
        total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

        dt_mean = np.mean(total_time)
        dt_std = np.std(total_time)
        freq_mean = 1 / dt_mean
        self.logger.info(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
        return freq_mean

    def record_closing_ceremony(self):
        closing_ceremony(self.master_bot_left, self.master_bot_right, self.env.puppet_bot_left, self.env.puppet_bot_right)
        self.reset_recording()

    def debug(self,):
        print(f'====== Debug mode ======')
        recorder = Recorder('right', is_debug=True)
        image_recorder = ImageRecorder(init_node=False, is_debug=True)
        while True:
            time.sleep(1)
            recorder.print_diagnostics()
            image_recorder.print_diagnostics()