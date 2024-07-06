import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import time
from torchvision import transforms

from constants import FPS
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos

from detr.models.latent_model import Latent_Model_Transformer

from sim_env import BOX_POSE
import subprocess

import IPython

def eval_bc(config, 
            is_sim = False,
            onscreen_render = False, 
            ckpt_name = 'policy_last.ckpt', 
            num_rollouts = 1, 
            fuse_real = False,
            reset_after_done = True,
            info_data = None):
    set_seed(config['seed'])
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    config['vq'] = False
    config['num_queries'] = config['chunk_size']
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std'] 
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    # load environment
    if not is_sim:
        #sys.path.append("..")
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True, setup_robots=True, setup_base=False,is_record=True)
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward
        #sys.path.append("..")
        if fuse_real:
            from aloha_scripts.robot_utils import move_grippers # requires aloha
            from aloha_scripts.real_env import make_real_env # requires aloha
            real_env = make_real_env(init_node=True, setup_robots=True, setup_base=False,is_record=True)

    query_frequency = config['chunk_size']
    if temporal_agg:
        #print("temporal aggregation is used")
        query_frequency = 1
        num_queries = config['num_queries']
    if not is_sim:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    opposite_dir = False
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
            #opposite_dir = True
            if not is_sim:
                camera_names = ['cam_high'] 
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()
        if fuse_real:
            real_ts = real_env.reset()

        ### onscreen render
        if onscreen_render and is_sim:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, config['action_dim']]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0 
            for t in range(max_timesteps):
                time1 = time.time()
                ### update onscreen render and wait for DT
                if onscreen_render and is_sim:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                time2 = time.time()
                obs = ts.observation
                if fuse_real:
                    real_obs = real_ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                #qpos_numpy = np.array(obs['qpos'])
                #use real robot qpos in sim_env
                if fuse_real:
                    qpos_numpy = np.array(real_obs['qpos'])
                    #if config['opposite_dir']:
                        #qpos_numpy[0] += 1.6
                        #qpos_numpy[7] -= 1.6
                else:
                    qpos_numpy = np.array(obs['qpos'])
                qpos_history_raw[t] = qpos_numpy
                qpos = pre_process(qpos_numpy)
                
                #if config['opposite_dir']:
                    #qpos[0][0] += 1.6
                    #qpos[0][7] -= 1.6
                #plt.ion()
                # qpos_history[:, t] = qpos
                if t % query_frequency == 0:
                    curr_image = get_image(ts, camera_names, rand_crop_resize= False)
                    
                    '''
                    l_cam = curr_image[0,1,:,:,:].cpu().numpy()
                    #reshape l_cam from 3*480*640 to 480*640*3
                    l_cam = np.transpose(l_cam, (1,2,0))
                    #change from BGR to RBG
                    l_cam = l_cam[...,::-1]
                    #change type to unit8
                    l_cam = (l_cam*255).astype(np.uint8)
                    l_cam = transforms.ToPILImage()(l_cam).convert("RGB")
                    #display image
                    plt.imshow(l_cam)
                    plt.draw()
                    plt.pause(0.001)
                    #print(curr_image.shape)
                    #curr_image = torch.zeros([1,1,3,480,640]).cuda()
                # print('get image: ', time.time() - time2)
                plt.ioff()
                '''
                info_data = obs
                info_data['qpos'] = qpos
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                if t == 0:
                    # warm up
                    for _ in range(10):
                        policy(qpos, curr_image)
                    print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                    # if use_actuator_net:
                    #     collect_base_action(all_actions, norm_episode_all_base_actions)
                    #if real_robot:
                        #all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                        #print(all_actions[:, :-BASE_DELAY, :-2].shape, all_actions[:, BASE_DELAY:, -2:].shape) #(ljy)
                if temporal_agg:
                    #if real_robot:
                        #if num_queries >= 87:
                            #num_queries = 87
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
                    # if t % query_frequency == query_frequency - 1:
                    #     # zero out base actions to avoid overshooting
                    #     raw_action[0, -2:] = 0
                
                # print('query policy: ', time.time() - time3)
                ### post-process actions
                time4 = time.time()
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]
                # if use_actuator_net:
                #     assert(not temporal_agg)
                #     if t % prediction_len == 0:
                #         offset_start_ts = t + history_len
                #         actuator_net_in = np.array(norm_episode_all_base_actions[offset_start_ts - history_len: offset_start_ts + future_len])
                #         actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
                #         pred = actuator_network(actuator_net_in)
                #         base_action_chunk = actuator_unnorm(pred.detach().cpu().numpy()[0])
                #     base_action = base_action_chunk[t % prediction_len]
                # else:
                base_action = action[-2:]
                # base_action = calibrate_linear_vel(base_action, c=0.19)
                # base_action = postprocess_base_action(base_action)
                # print('post process: ', time.time() - time4)
                
                info_data['target_qpos'] = target_qpos
                info_data['base_action'] = base_action
                ### step the environment
                time5 = time.time()
                if not is_sim:
                    #print(target_qpos)
                    ts = env.step(target_qpos, base_action)
                else:
                    #print(target_qpos)
                    ts = env.step(target_qpos)
                    if fuse_real:
                        real_ts = real_env.step(target_qpos,opposite_act=opposite_dir)
                # print('step env: ', time.time() - time5)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                # print(sleep_time)
                time.sleep(sleep_time)
                # time.sleep(max(0, DT - duration - culmulated_delay))
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    #print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
                # else:
                #     culmulated_delay = max(0, culmulated_delay - (DT - duration))

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
        #real_robot = True
        if not is_sim:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            #from aloha_scripts.record_episodes import closing_ceremony
            #closing_ceremony(env.puppet_bot_left, env.puppet_bot_right)
            # save qpos_history_raw
            log_id = get_auto_index(ckpt_dir)
            np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)
            plt.figure(figsize=(10, 20))
            # plot qpos_history_raw for each qpos dim using subplots
            for i in range(state_dim):
                plt.subplot(state_dim, 1, i+1)
                plt.plot(qpos_history_raw[:, i])
                # remove x axis
                if i != state_dim - 1:
                    plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f'qpos_{log_id}.png'))
            plt.close()

        if is_sim:
            rewards = np.array(rewards)
            episode_return = np.sum(rewards[rewards!=None])
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
    if is_sim:
        success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
        avg_return = np.mean(episode_returns)
        summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
        for r in range(env_max_reward+1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()
            more_or_equal_r_rate = more_or_equal_r / num_rollouts
            summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

        print(summary_str)

        # save success rate to txt
        result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
        with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
            f.write(summary_str)
            f.write(repr(episode_returns))
            f.write('\n\n')
            f.write(repr(highest_rewards))
            return success_rate, avg_return
    if reset_after_done:
        from reset_api import closing_ceremony
        closing_ceremony(env.puppet_bot_left, env.puppet_bot_right)
        

def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        #print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    
    return curr_image

def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")