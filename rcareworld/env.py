from pyrcareworld.envs import RCareWorld

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

        # Create robot object
        self.robot = self.create_robot(
            id=639787, 
            gripper_list=["6397870"], 
            robot_name="franka_panda",
            base_pos=[0, 0, 0],
        )

        #Pranav TODO : Insert any obstacles here for test env!!!

        print("Initialized RCareStorm object!")
        
    def set_robot_joint_positions(self, q):
        self.robot.setJointPositions(q)
        self._step()