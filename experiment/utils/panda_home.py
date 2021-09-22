import time
import numpy as np
import utils.panda_demo as utils

"""
 * a minimal script for teleoperating the robot using a joystick
 * Dylan Losey, September 2020

 * To run:
 [1] in one terminal:
    navigate to ~/panda-ws/essentials
    run python3 teleop.py
 [2] in a second terminal:
    navigate to ~/libfranka/build
    run ./collab/velocity_control
"""

home1 = np.asarray([1.859800e-02, -4.097640e-01,  1.386300e-02, -2.745454e+00, 2.011000e-03,  2.363305e+00, -7.419260e-01]) # Original Home

def return_home(conn, home):
    print('[*] Returning to Home!')
    total_time = 35.0
    start_time = time.time()
    dist = 1
    elapsed_time = time.time() - start_time
    waypoint = [0, 0, 0]
    at_waypoint = False
    dist_way = 1
    state = utils.readState(conn)
    s = np.asarray(utils.joint2pose(state['q']))
    while dist > 0.02 and elapsed_time < total_time:
        if np.array_equal(np.asarray(home), np.asarray(home1)) and not at_waypoint:
            if s[2] < 0.48:
                waypoint = [0.41, -0.4, 0.225]
            else:
                waypoint = [0.41, -0.4, 0.613]
            while dist_way > 0.02 and elapsed_time < total_time:
                state = utils.readState(conn)
                s = np.asarray(utils.joint2pose(state['q']))
                xdot = np.pad(np.clip(waypoint - s, -0.2, 0.2), (0, 3), 'constant')
                qdot = utils.xdot2qdot(xdot, state)
                utils.send2robot(conn, qdot)
                dist_way = np.linalg.norm(s - waypoint)
                elapsed_time = time.time() - start_time
            at_waypoint = True
        state = np.asarray(utils.readState(conn)['q'].tolist())
        qdot = np.clip(home - state, -0.2, 0.2)
        utils.send2robot(conn, qdot)
        dist = np.linalg.norm(state - home)
        elapsed_time = time.time() - start_time
    print('[*] Returned to Home!')

def main(conn):

    total_time = 40.0
    start_time = time.time()
    dist = 1
    elapsed_time = time.time() - start_time

    while dist > 0.02 and elapsed_time < total_time:
        state = np.asarray(utils.readState(conn)['q'].tolist())
        qdot = np.clip(home1 - state, -0.1, 0.1)
        utils.send2robot(conn, qdot)
        dist = np.linalg.norm(state - home1)
        elapsed_time = time.time() - start_time
    # print('[*] Done!')


if __name__ == "__main__":
    main()
