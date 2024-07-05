import os
import time
import h5py
import argparse
import h5py_cache
import numpy as np
from tqdm import tqdm
import cv2
from constants import DT, START_ARM_POSE, TASK_CONFIGS, FPS
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action
import rospy
from interbotix_xs_modules.arm import InterbotixManipulatorXS

import IPython
e = IPython.embed

def closing_ceremony(puppet_bot_left, puppet_bot_right,master_bot_left=None, master_bot_right=None):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    if master_bot_left == None:     #replay dont need master
        puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
        puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
        puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")

        puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
        puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
        puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")

        torque_on(puppet_bot_left)
        torque_on(puppet_bot_right)

                 #手动终端调试机械臂复位：
        CLOSE_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0]
        close_arm_qpos = CLOSE_ARM_POSE[:6]
                # 获取用户输入的6个值
        move_arms([ puppet_bot_left, puppet_bot_right], [close_arm_qpos] * 2, move_time=1.5)


        CLOSE_ARM_POSE = [0, -1.76, 1.46, 0, 0.2, 0]
        close_arm_qpos = CLOSE_ARM_POSE[:6]
        move_arms([ puppet_bot_left, puppet_bot_right], [close_arm_qpos] * 2, move_time=1.5)
                # move grippers to starting position
        move_grippers([puppet_bot_left,puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)
        torque_off(puppet_bot_left)
        torque_off(puppet_bot_right)

    else:

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

                        # replay only need puppet arm
            #手动终端调试机械臂复位：
        CLOSE_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0]
        close_arm_qpos = CLOSE_ARM_POSE[:6]
                # 获取用户输入的6个值
        move_arms([ puppet_bot_left, puppet_bot_right], [close_arm_qpos] * 4, move_time=1.5)


        CLOSE_ARM_POSE = [0, -1.76, 1.46, 0, 0.2, 0]
        close_arm_qpos = CLOSE_ARM_POSE[:6]
        move_arms([ puppet_bot_left, puppet_bot_right], [close_arm_qpos] * 4, move_time=1.5)
                # move grippers to starting position
        move_grippers([puppet_bot_left,puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 4, move_time=0.5)
        torque_off(puppet_bot_left)
        torque_off(puppet_bot_right)
        torque_off(master_bot_left)
        torque_off(master_bot_right)
        '''
    else:                           # record need master and puppet 
        CLOSE_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0]
        close_arm_qpos = CLOSE_ARM_POSE[:6]
            # 获取用户输入的6个值
        move_arms([ master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [close_arm_qpos] * 4, move_time=1.5)


        CLOSE_ARM_POSE = [0, -1.76, 1.46, 0, 0.2, 0]
        close_arm_qpos = CLOSE_ARM_POSE[:6]
        move_arms([ master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [close_arm_qpos] * 4, move_time=1.5)

        # move grippers to starting position
        move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)

        torque_off(puppet_bot_left)
        torque_off(puppet_bot_right)
        torque_off(master_bot_left)
        torque_off(master_bot_right)
        '''

                    # 定义一个索引，表示当前选定的值
    '''manual mode to adjust the CLOSE_ARM_POSE:
    index = 0
    
    while True:

        close_arm_qpos = CLOSE_ARM_POSE[:6]
        move_arms([ puppet_bot_left, puppet_bot_right], [close_arm_qpos] * 2, move_time=1.5)
        # move grippers to starting position
        move_grippers([puppet_bot_left,puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)

        # 打印当前的CLOSE_ARM_POSE和选定的值
        print("CLOSE_ARM_POSE:", CLOSE_ARM_POSE)
        print("当前选定的值:", CLOSE_ARM_POSE[index])

        # 获取用户输入
        user_input = input("按'a'增加值，按'd'减少值，按'<'选择上一个值，按'>'选择下一个值，按'q'退出：")

        # 根据用户输入调整值
        if user_input == 'a':
            CLOSE_ARM_POSE[index] += 0.1
        elif user_input == 'd':
            CLOSE_ARM_POSE[index] -= 0.1
        elif user_input == '<':
            index = (index - 1) % len(CLOSE_ARM_POSE)
        elif user_input == '>':
            index = (index + 1) % len(CLOSE_ARM_POSE)
        elif user_input == 'q':
            break
        else:
            print("无效的输入。")
'''
    
        


def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right,episode_end=False):
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
        pressed = False
        while not pressed:
            gripper_pos_left = get_arm_gripper_positions(master_bot_left)
            gripper_pos_right = get_arm_gripper_positions(master_bot_right)
            if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
                pressed = True
            time.sleep(DT/10)
        torque_off(master_bot_left)
        torque_off(master_bot_right)

    # right_gripper_info = puppet_bot_right.dxl.srv_get_info("single", "gripper")
    # left_gripper_info = puppet_bot_left.dxl.srv_get_info("single", "gripper")
    # print('\npuppet bot griper mode, left:{}, right:{}\n'.format(left_gripper_info.mode, right_gripper_info.mode))

    print(f'Started!')


def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    print(f'Dataset name: {dataset_name}')

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

    

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)
    # Open puppet grippers
    env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

   

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

   
    # plot /base_action vs /base_action_t265
    # import matplotlib.pyplot as plt
    # plt.plot(np.array(data_dict['/base_action'])[:, 0], label='base_action_linear')
    # plt.plot(np.array(data_dict['/base_action'])[:, 1], label='base_action_angular')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 0], '--', label='base_action_t265_linear')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 1], '--', label='base_action_t265_angular')
    # plt.legend()
    # plt.savefig('record_episodes_vel_debug.png', dpi=300)


    

    
    
    return True


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite)
        if is_healthy:
            break

        env = make_real_env(init_node=True, setup_base=False,is_record=False)
        env.reset()
        closing_ceremony(env.puppet_bot_left, env.puppet_bot_right,env.master_bot_left,env.master_bot_right)


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    
    main(vars(parser.parse_args())) # TODO
    # debug()


