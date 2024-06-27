import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize MuJoCo 
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
t2 = 2
t3 = 1
total_time = t1 + t2 + t3
v = -1 # towards negative x-direction
theta = 0 
phi = 0

simend = total_time + 1 
step = 100
average_time_step = total_time / step
# Result storage
jointA_position = []; jointB_position = []; joint_position = []
jointA_velocity = []; jointB_velocity = []; joint_velocity = []
jointA_qacc = []; jointB_qacc = []; joint_qacc = []
dot_x = np.array([]); dot_v = np.array([]); real_dotx = []
theta_numerial = np.array([]); phi_numerial = np.array([])
sim_time = []; time_i =[]
jointA_torque = []; jointB_torque = []
history_result = []
i = 0; j = 0

def compute_x_v(t_stamp): #cal all x(t_stamp)
    if 0 <= t_stamp <= t1:
        x = L1 + L2 + 0.5 * (v / t1) * t_stamp ** 2
        current_v = v/t1*t_stamp
    elif t1 < t_stamp <= (t1 + t2):
        x = L1 + L2 + v * (0.5 * t1 + t_stamp - t1)
        current_v = v
    elif (t1 + t2) < t_stamp <= total_time:
        x = L1 + L2 + v * (0.5 * t1 + t2) + 1/2*((v / t3)*(total_time-t_stamp)+v)*(t_stamp-t1-t2)
        current_v = v/t3*(total_time - t_stamp)
    else:
        x = L1 + L2 + v * (0.5 * t1 + t2 + 0.5 * t3)
        current_v = 0
    return x, current_v

def compute_angles(x): #theta & phi shoudl be +/- opposite
    theta = np.arccos((x ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2))
    phi = -1 * np.arctan(L2 * np.sin(theta) / (L1 + L2 * np.cos(theta)))
    return theta, phi

def compute_torques(theta, phi):
    # data.qfrc_applied[model.joint_name2id('joint_A')] = mass
    # omega = theta - phi
    # T2 = -mass * g * L2 * np.cos(theta)
    # T1 = 
    # return T1, T2
    pass

def compute_angularvelocities(x, theta, phi, current_v):
    if np.sin(theta) == 0:
        theta_dot = 0
    else:
        theta_dot = -1/(L1*L2)*x*current_v/np.sin(theta)
    
    phi_dot = (L1*L2*np.cos(theta)*theta_dot + L2**2*np.cos(theta)**2*theta 
               - L2**2*np.sin(theta)**2*theta_dot)*np.cos(phi)**2/(L1+L2*np.cos(theta))**2
    return theta_dot, phi_dot

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def controller(model, data, t_stamp):
    global dot_x, dot_v, theta_numerial, phi_numerial #np array
    x, current_v = compute_x_v(t_stamp)
    theta, phi = compute_angles(x)
    theta_dot, phi_dot = compute_angularvelocities(x, theta, phi, current_v)

    dot_x = np.append(dot_x, x)
    dot_v = np.append(dot_v, current_v)
    theta_numerial = np.append(theta_numerial, theta_dot)
    phi_numerial = np.append(phi_numerial, phi_dot) 

    # data.ctrl[0] = phi
    # data.ctrl[1] = theta
    # data.qpos[0] = phi
    # data.qpos[1] = theta

    # data.qvel[0] = phi_dot
    # data.qvel[1] = theta_dot 
    # prev_theta_dot = theta_dot
    # prev_phi_dot = phi_dot

    if phi < -np.pi/2 or phi > np.pi/2:
        print("Phi out of range:", phi)
        return True
    return False


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

# data.time = 0.0 now
while not glfw.window_should_close(window):
    time_prev = data.time
    i+=1; time_i.append(time_prev)
    
    while (data.time - time_prev < 1.0/60.0): # Step the simulation every 0.002 second
        mj.mj_step(model, data)
        sim_time.append(data.time)
        # stop = controller(model, data, data.time)
        joint_position.append(data.qpos.copy())
        jointA_position.append(joint_position[j][0])
        jointB_position.append(joint_position[j][1])
        joint_velocity.append(data.qvel.copy())
        jointA_velocity.append(joint_velocity[j][0])
        jointB_velocity.append(joint_velocity[j][1])
        joint_qacc.append(data.qacc.copy())
        jointA_qacc.append(joint_qacc[j][0])
        jointB_qacc.append(joint_qacc[j][1])
        data.qfrc_applied[1] = mass*L2**2 * joint_qacc[j][1] + mass*g*L2*np.cos(joint_position[j][1]-joint_position[j][0])
        data.qfrc_applied[0] = mass*L2**2 * joint_qacc[j][0] - mass*g*(L1*np.cos(joint_position[j][0])+L2*np.cos(joint_position[j][1]-joint_position[j][0])) 
        jointA_torque.append(data.joint("joint_A").qfrc_constraint + data.joint("joint_A").qfrc_smooth)
        jointB_torque.append(data.joint("joint_B").qfrc_constraint + data.joint("joint_B").qfrc_smooth)
        real_dotx.append(data.site_xpos[0][0])
        j+=1
    #     if stop:
    #         break  # Break the loop if phi is out of range
    # if stop:
    #     break  # Break the outer loop if phi is out of range
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
real_dotx = np.array(real_dotx)
# print("Size of jointA_position:", len(jointA_position))
# print("Size of jointB_position:", len(jointB_position))
# print("time_i:", time_i); print(sim_time)
print("Final position of the dot:", data.site_xpos[0])
print(data.joint('joint_A'))
# print(data.site('dot'))

_, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 10))
ax[0,0].plot(sim_time, jointB_position)
ax[0,0].set_title('theta (middle joint)')
ax[0,0].set_ylabel('radian')

ax[1,0].plot(sim_time, jointB_velocity)
ax[1,0].set_title('angular velocity (theta)')
ax[1,0].set_ylabel('radian/second?')

ax[2,0].plot(sim_time, jointB_qacc)
ax[2,0].set_title('theta acceleration')
ax[2,0].set_ylabel('radian/second^2')
ax[2,0].set_xlabel('second')

ax[0,1].plot(sim_time, jointA_position)
ax[0,1].set_title('phi (leftmost joint)')
ax[0,1].set_ylabel('radian')

ax[1,1].plot(sim_time, jointA_velocity)
ax[1,1].set_title('angular velocity (phi)')
ax[1,1].set_ylabel('radian/second')

ax[2,1].plot(sim_time, jointA_qacc)
ax[2,1].set_title('phi acceleration')
ax[2,1].set_ylabel('radian/second^2')
ax[2,1].set_xlabel('second')

plt.tight_layout()
plt.show()

_, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 10))
ax[0,0].plot(sim_time, dot_x)
ax[0,0].set_title('expected x position of the dot')
ax[0,0].set_ylabel('meter')

ax[1,0].plot(sim_time, theta_numerial)
ax[1,0].set_title('expected angular velocity (theta)')
ax[1,0].set_ylabel('radian/second')

ax[2,0].plot(sim_time, jointB_torque)
ax[2,0].set_title('torque of middle joint at the simulation')
ax[2,0].set_ylabel('Newton meter(?)')
ax[2,0].set_xlabel('second')

ax[0,1].plot(sim_time, dot_v)
ax[0,1].set_title('expected v')
ax[0,1].set_ylabel('meter/second')

ax[1,1].plot(sim_time, phi_numerial)
ax[1,1].set_title('expected angular velocity (phi)')
ax[1,1].set_ylabel('radian/second')

ax[2,1].plot(sim_time, jointA_torque)
ax[2,1].set_title('torque of leftmost joint at the simulation')
ax[2,1].set_ylabel('Newton meter(?)')
ax[2,1].set_xlabel('second')

plt.tight_layout()
plt.show()