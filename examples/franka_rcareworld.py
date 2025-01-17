#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
""" Example spawning a robot in gym 

"""
import copy
# from isaacgym import gymapi
# from isaacgym import gymutil

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#



import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

# from storm_kit.gym.core import Gym, World
# from storm_kit.gym.sim_robot import RobotSim
# from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
# from storm_kit.gym.helpers import load_struct_from_dict
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask

from scipy.spatial.transform import Rotation as R

from rcareworld.env import RCareStorm
from rcareworld.utils import unity2storm

np.set_printoptions(precision=2)

def mpc_robot_interactive(args, sim_params):

    colorama_init()

    vis_ee_target = True
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_rcare_reacher.yml'
    # world_file = 'collision_primitives_3d.yml'               # use when replicating isaac sim env
    world_file = 'collision_empty.yml'

    env = RCareStorm()

    if(args.cuda):
        device = 'cuda'
    else:
        device = 'cpu'

    device = torch.device('cuda', 0) 

    tensor_args = {'device':device, 'dtype':torch.float32}

    # get camera data:
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    
    mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}

    tgt_p = env.get_target_eef_pose()
    g_pos = unity2storm(position=tgt_p['position'])
    g_q    = unity2storm(orientation=tgt_p['orientation']) 
    
    mpc_control.update_params(goal_ee_pos=g_pos, goal_ee_quat=g_q)

    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    
    rollout = mpc_control.controller.rollout_fn
    tensor_args = mpc_tensor_dtype
    sim_dt = mpc_control.exp_params['control_dt']
    
    log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                'qddd_des':[]}

    qd_des = np.zeros(env.robot_dof)
    t_step = 0

    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

    t_now = time.time()

    qdd_des = np.array([0,0,0,0,0,0,0])
    # env.set_robot_joint_position(joint_positions=(np.array([-0.9693274502997367, 0.9422471412712752, 0.7439066754002944, 1.019769135341833, 1.672839590475963, -1.3861354591630555])*180/np.pi))

    ii = 0
    time.sleep(5)

    while True:
        try:
            # if(ii > 230):
            #     env.set_robot_joint_position(joint_positions=q_des)
            #     env.step()
            #     continue
            
            #Find goal position
            tgt_p = env.get_target_eef_pose()
            # print(tgt_p)

            #update goal position only if it has moved more than a threshold
            if(np.linalg.norm(g_pos - unity2storm(position=tgt_p['position'])) > 0.00001) or (np.linalg.norm(g_q - unity2storm(orientation=tgt_p['orientation']))>0.0001):
                g_pos = unity2storm(position=tgt_p['position'])
                g_q   = unity2storm(orientation=tgt_p['orientation'])
                
                mpc_control.update_params(goal_ee_pos=g_pos,
                                              goal_ee_quat=g_q)
    
            #update timestep
            t_step += sim_dt

            # print("Input Goal Position:",g_pos)
            # print("Input Goal Orientation:",storm_q)
            
            q_des = (np.radians(np.array(env.get_robot_joint_positions()) % 360.))
            qd_des = (np.radians(np.array(env.get_robot_joint_velocities()) % 360.))
            for i in range(7):
                if q_des[i] > np.pi:
                    q_des[i] = q_des[i] - (2 * np.pi) # -pi, pi
            
            current_robot_state = {}
           
            current_robot_state['position'] = copy.deepcopy(q_des)
            current_robot_state['velocity'] = copy.deepcopy(qd_des)
            # current_robot_state['acceleration'] = copy.deepcopy(qdd_des)
            current_robot_state['acceleration'] = np.zeros(n_dof)
        
            command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=False)
            
            filtered_state_mpc = current_robot_state 
            curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            # get position command:
            q_des = copy.deepcopy(command['position'])
            qd_des = copy.deepcopy(command['velocity']) #* 0.5
            qdd_des = copy.deepcopy(command['acceleration'])
            
            ee_error = mpc_control.get_current_error(filtered_state_mpc)
             
            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)

            if (ii > 200) :
                print(ii)
                print(f"{Fore.CYAN}Target eef posn:",g_pos)
                print(f"Target eef orn:",g_q)
                print(f"Curr eef posn:",pose_state["ee_pos_seq"])
                print(f"Curr eef orn :",pose_state["ee_quat_seq"])
                print(f"--------------------------------------------------------")
                print(f"ee_error: ",ee_error)
                print(f"--------------------------------------------------------")
            else:
                print(ii)
                print("Target eef posn:",g_pos)
                print("Target eef orn:",g_q)
                print("Curr eef posn:",pose_state["ee_pos_seq"])
                print("Curr eef orn :",pose_state["ee_quat_seq"])
                print("--------------------------------------------------------")
                print("ee_error: ",ee_error)
                print("--------------------------------------------------------")

            # # print(np.degrees(q_des))
            # print(np.radians(env.get_robot_joint_positions()))
            # print("--------------------------------------------------------")

            q_des = np.degrees(q_des)
           
            t_now = time.time()
            if(ii > 200):
               env.set_robot_joint_position(joint_positions=q_des)
            env.step()

            current_state = command
            ii += 1
        except KeyboardInterrupt:
            print('Closing')

            done = True
            break

    mpc_control.close()
    return 1 
    
if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    # gym_instance = Gym(**sim_params)
    
    
    mpc_robot_interactive(args, sim_params)
    
