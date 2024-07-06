import os
import h5py
import numpy as np
from aloha_scripts.robot_utils import move_grippers, calibrate_linear_vel, smooth_base_action, postprocess_base_action
import argparse
import matplotlib.pyplot as plt
from aloha_scripts.real_env import make_real_env
from aloha_scripts.constants import JOINT_NAMES, PUPPET_GRIPPER_JOINT_OPEN, FPS
import time
from aloha_scripts.constants import DT, START_ARM_POSE, TASK_CONFIGS, FPS
from aloha_scripts.constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from aloha_scripts.robot_utils import move_arms, torque_on, torque_off, move_grippers

from interbotix_xs_modules.arm import InterbotixManipulatorXS
#import sys
#import osu00
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import IPython
e = IPython.embed

STATE_NAMES = JOINT_NAMES + ["gripper", 'left_finger', 'right_finger']
 
def close_arm(close_master = False):
    
    
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",   #激活两个主动端手爪
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False, is_record=True)  #激活两个从动端手爪
    env.reset()

    #划分了两个任务：（1）close_puppet:只需要关闭从动端手抓（用于replay和imitate_episodes的评估任务）；（2）关闭主动和从动端:需要关闭主动端和从动端手爪（用于record任务）
    if not close_master:
        closing_ceremony(master_bot_left=None, master_bot_right=None,puppet_bot_left=env.puppet_bot_left, puppet_bot_right=env.puppet_bot_right)
    else:
    #record mode :use both master and puppet
        
        closing_ceremony(master_bot_left=master_bot_left, master_bot_right=master_bot_right,puppet_bot_left=env.puppet_bot_left, puppet_bot_right=env.puppet_bot_right)

def closing_ceremony(puppet_bot_left, puppet_bot_right,master_bot_left=None, master_bot_right=None):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    if master_bot_left == None:     #replay dont need master

        torque_on(puppet_bot_left)
        torque_on(puppet_bot_right)

                 #手动终端调试机械臂复位：
        CLOSE_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0]
        close_arm_qpos = CLOSE_ARM_POSE[:6]
               
        move_arms([ puppet_bot_left, puppet_bot_right], [close_arm_qpos] * 2, move_time=1.5)

        time.sleep(1.5)
        PUPPET_CLOSE_ARM_POSE = [0, -1.8600000000000008, 1.5600000000000003, 0, 0.7, 0]
        close_arm_qpos = PUPPET_CLOSE_ARM_POSE[:6]
        move_arms([ puppet_bot_left, puppet_bot_right], [close_arm_qpos] * 2, move_time=1.5)
                # move grippers to starting position
        move_grippers([puppet_bot_left,puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)
        time.sleep(1.5)
        torque_off(puppet_bot_left)
        torque_off(puppet_bot_right)
    
    else:

        master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
        # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

        torque_on(puppet_bot_left)
        torque_on(master_bot_left)
        torque_on(puppet_bot_right)
        torque_on(master_bot_right)

                       
            #手动终端调试机械臂复位：
        CLOSE_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0]
        close_arm_qpos = CLOSE_ARM_POSE[:6]
                # 获取用户输入的6个值
        move_arms([ master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [close_arm_qpos] * 4, move_time=1.5)

        time.sleep(1.5)
        CLOSE_ARM_POSE = [0, -1.76, 1.46, 0, 0.2, 0]
        PUPPET_CLOSE_ARM_POSE = [0, -1.8600000000000008, 1.5600000000000003, 0, 0.7, 0]
        MASTER_CLOSE_ARM_POSE = [0, -1.8600000000000008, 1.5600000000000003, 0, 0.30000000000000004, 0]

        puppet_close_arm_qpos = PUPPET_CLOSE_ARM_POSE[:6]
        move_arms([  puppet_bot_left,  puppet_bot_right], [puppet_close_arm_qpos] * 2, move_time=1.5)
        time.sleep(1.5)
        master_close_arm_qpos = MASTER_CLOSE_ARM_POSE[:6]
        move_arms([ master_bot_left,  master_bot_right], [master_close_arm_qpos] * 2, move_time=1.5)


        # move grippers to starting position
        move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 4, move_time=0.5)
        time.sleep(1.5)
        
        torque_off(puppet_bot_left)
        torque_off(puppet_bot_right)
        torque_off(master_bot_left)
        torque_off(master_bot_right)
def main(args):
    close_arm(args['close_puppet'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--close_puppet',default=True, action='store_true', help='only_close_puppet.', required=False)
    main(vars(parser.parse_args()))


