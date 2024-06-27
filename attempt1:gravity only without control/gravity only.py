import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize MuJoCo model and data
xml_path = '2D_robotic_arm.xml'
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera() 
opt = mj.MjvOption() 

# Control variables
L1 = 3.0
L2 = 7.0
mass = 1.0
g = 9.81
# Motion parameters
t1 = 1
t2 = 3
t3 = 5
total_time = t1 + t2 + t3
v = -0.1 # negative x-direction
theta = 0 
phi = 0

simend = 1 + total_time + 1 
step = 100
average_time_step = total_time / step
# Result storage
jointA_position = []; jointB_position = []; joint_position = []
jointA_velocity = []; jointB_velocity = []; joint_velocity = []
jointA_qacc = []; jointB_qacc = []; joint_qacc = []
sim_time = []
jointA_torque = []; jointB_torque = []
history_result = []
i = 0; j = 0

def compute_x(t_stamp): #cal all x(t_stamp)
    if 0 <= t_stamp <= t1:
        x = L1 + L2 + 0.5 * (v / t1) * t_stamp ** 2
    elif t1 < t_stamp <= (t1 + t2):
        x = L1 + L2 + v * (0.5 * t1 + t_stamp)
    elif (t1 + t2) < t_stamp <= total_time:
        x = L1 + L2 + v * (0.5 * t1 + t1 + t2) - (v / t3) * (0.5 * (t1 + t2) ** 2 - (t1 + t2) * total_time
                                                             + total_time * t_stamp - 0.5 * t_stamp ** 2)
    else:
        x = L1 + L2 + v * (0.5 * t1 + t1 + t2 + 0.5 * t3)
    return x

def compute_angles(x): #theta & phi shoudl be +/- opposite
    theta = np.arccos((x ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2))
    phi = np.arctan(L2 * np.sin(theta) / (L1 + L2 * np.cos(theta))) 
    return theta, phi

def compute_torques(theta, phi):
    pass

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def controller(model, data, t_stamp):
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# camera configuration
cam.azimuth = 91.19999999999999 
cam.elevation = -85.79999999999995 
cam.distance =  29.393051762183
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

mj.set_mjcb_control(lambda model, data: controller(model, data, data.time))

# data.time= 0.0 now
while not glfw.window_should_close(window):
    time_prev = data.time
    i+=1
    
    while (data.time - time_prev < 1.0/60.0): # Step the simulation after 1/60 second
        mj.mj_step(model, data)
        sim_time.append(data.time)
        joint_position.append(data.qpos.copy())
        jointA_position.append(joint_position[j][0])
        jointB_position.append(joint_position[j][1])
        joint_velocity.append(data.qvel.copy())
        jointA_velocity.append(joint_velocity[j][0])
        jointB_velocity.append(joint_velocity[j][1])
        joint_qacc.append(data.qacc.copy())
        jointA_qacc.append(joint_qacc[j][0])
        jointB_qacc.append(joint_qacc[j][1])
        jointA_torque.append(data.joint("joint_A").qfrc_constraint + data.joint("joint_A").qfrc_smooth)
        jointB_torque.append(data.joint("joint_B").qfrc_constraint + data.joint("joint_B").qfrc_smooth)
        j+=1
    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()


print("Simulated time:", time_prev)
print("Number of timestamp:", len(sim_time))

# Convert results to numpy array for easier analysis
history_result = np.array(history_result)
print(i, j)

# Convert to numpy arrays for plotting
sim_time = np.array(sim_time)
jointA_position = np.array(jointA_position)
jointB_position = np.array(jointB_position)
joint_position = np.array(joint_position)
print("Size of jointA_position:", len(jointA_position))
print("Size of jointB_position:", len(jointB_position))
jointA_torque = np.array(jointA_torque)
jointB_torque = np.array(jointB_torque)

_, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 10))
ax[0,0].plot(sim_time, jointB_position)
ax[0,0].set_title('theta (middle joint)')
ax[0,0].set_ylabel('radian')

ax[1,0].plot(sim_time, jointB_velocity)
ax[1,0].set_title('theta velocity')
ax[1,0].set_ylabel('radian/second?')

ax[2,0].plot(sim_time, jointB_qacc)
ax[2,0].set_title('theta acceleration')
ax[2,0].set_ylabel('radian/second^2')
ax[2,0].set_xlabel('second')

ax[0,1].plot(sim_time, jointA_position)
ax[0,1].set_title('phi (leftmost joint)')
ax[0,1].set_ylabel('radian')

ax[1,1].plot(sim_time, jointA_velocity)
ax[1,1].set_title('phi velocity')
ax[1,1].set_ylabel('radian/second')

ax[2,1].plot(sim_time, jointA_qacc)
ax[2,1].set_title('phi acceleration')
ax[2,1].set_ylabel('radian/second^2')
ax[2,1].set_xlabel('second')
plt.tight_layout()
plt.show()
