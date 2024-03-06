## Install Instructions

### System Dependencies:
- Conda version >= 4.9
- NVIDIA driver >= 460.32
- Cuda toolkit >= 11.0

Steps:

1. Create a new conda environment with: conda env create -f environment.yml

2. Install python bindings for isaacgym: https://developer.nvidia.com/isaac-gym

3. Run the following command from this directory: pip install -e . 

If you are using STORM with RCareWorld, proceed with the following instructions 

4. Follow installation steps for RCareWorld : [link](https://sunny-building-1e7.notion.site/Install-the-Modules-014832fa7896462893ce332cf8e70a7f)

5. Run the following command in your terminal, to paste the Unity scene file to your Unity project :
    ```cp  unity/franka_only.unity <your_unity_project>/Assets/RCareWorld/Scenes```

6. 



### Running Example

1. Run scripts/train_self_collision.py to get weights for robot self collision checking : 
    ``` python scripts/train_self_collision.py ```

2. Run : ``` python examples/franka_rcareworld.py ```

3. Press play in the Unity main window

4. Move the reference cube (id : 30001) around in the view and watch the robot end effector track the cube pose. 

