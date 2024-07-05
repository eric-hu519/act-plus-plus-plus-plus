import os
import h5py
import numpy as np
from robot_utils import move_grippers, calibrate_linear_vel, smooth_base_action, postprocess_base_action
import argparse
import matplotlib.pyplot as plt
from real_env import make_real_env
from constants import JOINT_NAMES, PUPPET_GRIPPER_JOINT_OPEN, FPS
import time
from constants import  TASK_CONFIGS
from record_episodes import get_auto_index, closing_ceremony
from interbotix_xs_modules.arm import InterbotixManipulatorXS
#import sys
#import osu00
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import robot_utils

import IPython
e = IPython.embed

STATE_NAMES = JOINT_NAMES + ["gripper", 'left_finger', 'right_finger']

def main(args):
    
    #master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
    #                                          robot_name=f'master_left', init_node=True)
    #master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
    #                                           robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=True, setup_base=False,is_record=False)
    env.reset()

    closing_ceremony(env.puppet_bot_left, env.puppet_bot_right)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=False)
    
    parser.add_argument('--task_name',default= 'aloha_mobile_dummy', action='store', type=str, help='Task name.', required=False)
    parser.add_argument('--episode_idx',default='3' , action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)
    parser.add_argument('--use_base',default=False, action='store_true', help='Use base actions.', required=False)
    
    main(vars(parser.parse_args()))


