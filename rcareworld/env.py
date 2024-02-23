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

        self.sphere_1 = env.create_object(id=10001, name="Sphere 1", is_in_scene=False)
        self.cube_1 = env.create_object(id=20001, name="Cube 1", is_in_scene=False)
        self.cube_2 = env.create_object(id=30002, name="Cube 2", is_in_scene=False)
        self.cube_3 = env.create_object(id=30003, name="Cube 3", is_in_scene=False)

        self.sphere_1.load()
        self.cube_1.load()
        self.cube_2.load()
        self.cube_3.load()

        #positions after coordinate transformation from IsaacSim to Unity
        #original position from collision_primitives_3d.yml
        #rotation quaternion does not change after rotation around axis
        self.sphere_1.setTransform(position=[-0.4, 0.1, 0.4],scale=[0.1, 0.1, 0.1])
        self.cube_1.setTransform(position=[-0.2, 0.2, 0.4],scale=[0.3, 0.1, 0.4])
        self.cube_2.setTransform(position=[0.3, 0.2, 0.4],scale=[0.3, 0.1, 0.5])
        self.cube_3.setTransform(position=[0.0, -0.1, 0.0],scale=[2.0, 2.0, 0.2])

        print("Initialized RCareStorm object!")
        
    def set_robot_joint_positions(self, q):
        self.robot.setJointPositions(q)
        self._step()