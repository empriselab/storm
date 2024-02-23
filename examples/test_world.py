from pyrcareworld.envs import RCareWorld

# Create robot object
env = RCareWorld()

robot_id = 639787
robot_dof = 7
robot = env.create_robot(
    id=robot_id, 
    gripper_list=["6397870"], 
    robot_name="franka_panda",
    base_pos=[0, 0, 0],
)

for i in range(100):
    env.step()

sphere_1 = env.create_object(id=10001, name="Sphere", is_in_scene=False)
sphere_1.load()
sphere_1.setTransform(position=[-0.4, 0.1, 0.4],scale=[0.1, 0.1, 0.1])

cube_1   = env.create_object(id=20001, name="Cube", is_in_scene=False)
cube_1.load()
cube_1.setTransform(position=[-0.2, 0.2, 0.4],scale=[0.3, 0.1, 0.4])

cube_2   = env.create_object(id=20002, name="Cube", is_in_scene=False)
cube_2.load()
cube_2.setTransform(position=[0.3, 0.2, 0.4],scale=[0.3, 0.1, 0.5])

cube_3   = env.create_object(id=20003, name="Cube", is_in_scene=False)
cube_3.load()
cube_3.setTransform(position=[0.0, -0.1, 0.0],scale=[2.0, 2.0, 0.2])

cube_ref = env.create_object(id=30001, name="Cube", is_in_scene=False)
cube_ref.load()
cube_ref.setTransform(position=[-0.3, 0.5, 0.0],scale=[0.01, 0.01, 0.01])

# #positions after coordinate transformation from IsaacSim to Unity
# #original position from collision_primitives_3d.yml
# #rotation quaternion does not change after rotation around axis




init_joint_state = [133.62, -27.56, -91.45, -42.93, 
                    -36.87, 49.87, 61.51]
env.instance_channel.set_action(
                "SetJointPositionDirectly",
                id=robot_id,
                joint_positions=list(init_joint_state)
            )

# env.stepSeveralSteps(100)
# robot.load()
while True:
    env.instance_channel.set_action(
                "SetJointPositionDirectly",
                id=robot_id,
                joint_positions=list(init_joint_state)
            )
    env._step()