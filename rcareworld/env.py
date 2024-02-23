from pyrcareworld.envs import RCareWorld
import pybullet as p
import numpy as np
import math

class RCareStorm(RCareWorld):
    
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        custom_channels: list = [],
        assets: list = [],
        **kwargs
    ):
        RCareWorld.__init__(
            self,
            executable_file=executable_file,
            scene_file=scene_file,
            custom_channels=custom_channels,
            assets=assets,
            **kwargs,
        )

        self.robot_id = 639787
        self.robot_dof = 7

        # Create robot object
        self.robot = self.create_robot(
            id=self.robot_id, 
            gripper_list=["6397870"], 
            robot_name="franka_panda",
            base_pos=[0, 0, 0],
        )

        init_joint_state = [1.00, -1.0, 0.00, -2.0, 0.00, 1.57, 0.78]
        self.robot.setJointPositionsDirectly(init_joint_state)

        self.sphere_1 = env.create_object(id=10001, name="Sphere 1", is_in_scene=False)
        self.cube_1 = env.create_object(id=20001, name="Cube 1", is_in_scene=False)
        self.cube_2 = env.create_object(id=20002, name="Cube 2", is_in_scene=False)
        self.cube_3 = env.create_object(id=20003, name="Cube 3", is_in_scene=False)

        self.cube_ref = env.create_object(id=30001, name="Cube 3", is_in_scene=False)

        self.sphere_1.load()
        self.cube_1.load()
        self.cube_2.load()
        self.cube_3.load()
        self.cube_ref.load()

        #positions after coordinate transformation from IsaacSim to Unity
        #original position from collision_primitives_3d.yml
        #rotation quaternion does not change after rotation around axis
        self.sphere_1.setTransform(position=[-0.4, 0.1, 0.4],scale=[0.1, 0.1, 0.1])
        self.cube_1.setTransform(position=[-0.2, 0.2, 0.4],scale=[0.3, 0.1, 0.4])
        self.cube_2.setTransform(position=[0.3, 0.2, 0.4],scale=[0.3, 0.1, 0.5])
        self.cube_3.setTransform(position=[0.0, -0.1, 0.0],scale=[2.0, 2.0, 0.2])

        self.cube_ref.setTransform(position=[-0.3, 0.5, 0.0],scale=[0.01, 0.01, 0.01])

        print("Initialized RCareStorm object!")

        self._step()

    def get_target_eef_pose(self):
        target_pose = {}
        target_pose['position'] = self.instance_channel.data[30001]['position']
        target_pose['orientation'] = self.instance_channel.data[30001]['rotation']
        return target_pose
        
    def set_robot_joints(self, joint_positions=None, joint_velocities=None, joint_accelerations=None):
        if joint_positions is not None:
            self.instance_channel.set_action(
                'SetJointPositionDirectly',
                id=self.robot_id,
                joint_positions=list(joint_positions[0:self.robot_dof]),
            )
        
        if joint_velocities is not None:
            self.instance_channel.set_action(
                'SetJointVelocity',
                id=self.robot_id,
                joint_velocitys=list(joint_velocities[0:self.robot_dof]),
            )

        if joint_accelerations is not None:
            self.instance_channel.set_action(
                    'SetJointAcceleration',
                    id=self.robot_id,
                    joint_accelerations=list(joint_accelerations[0:self.robot_dof]),
                )

        self._step()
    
    def set_eef_pose(self, pos: list, rot: list):
        q = self.robot.ik_controller.calculate_ik_recursive(pos, eef_orn=p.getQuaternionFromEuler(rot))
        self.set_robot_joints(joint_positions=joint_positions)

    def step(self):
        self._step()