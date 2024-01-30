#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import time
from draw import Draw_MPC_point_stabilization_v1


if __name__ == '__main__':
    T = 0.1
    N = 100
    av_max = 2.0
    wv_max = np.pi

    # control variables, linear acceleration av and angle acceletation aw for mobile base. aj for manipulator joints
    av = ca.SX.sym('av')
    aw = ca.SX.sym('aw')
    aj1 = ca.SX.sym('aj1')
    aj2 = ca.SX.sym('aj2')
    aj3 = ca.SX.sym('aj3')
    aj4 = ca.SX.sym('aj4')
    aj5 = ca.SX.sym('aj5')
    aj6 = ca.SX.sym('aj6')
    controls = ca.vertcat(av, aw, aj1, aj2, aj3, aj4, aj5, aj6)

    # state variables, x, y, theta, j1, j2, j3, j4, j5, j6, v, omega, j1_dot, j2_dot, j3_dot, j4_dot, j5_dot, j6_dot
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    j1 = ca.SX.sym('j1')    
    j2 = ca.SX.sym('j2')
    j3 = ca.SX.sym('j3')
    j4 = ca.SX.sym('j4')
    j5 = ca.SX.sym('j5')
    j6 = ca.SX.sym('j6')
    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    j1_dot = ca.SX.sym('j1_dot')
    j2_dot = ca.SX.sym('j2_dot')
    j3_dot = ca.SX.sym('j3_dot')
    j4_dot = ca.SX.sym('j4_dot')
    j5_dot = ca.SX.sym('j5_dot')
    j6_dot = ca.SX.sym('j6_dot')
    states = ca.vertcat(x, y, theta, j1, j2, j3, j4, j5, j6, v, omega, j1_dot, j2_dot, j3_dot, j4_dot, j5_dot, j6_dot)

    # create model
    STATE_DIM = 17
    INPUT_DIM = 8
    BASE_INPUT = 2
    ARM_INPUT = 6
    POSITION_STATE_DIM = 9
    VELOCITY_STATE_DIM = 8


    rhs = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta), omega, j1_dot, j2_dot, j3_dot, j4_dot, j5_dot, j6_dot,
                     av, aw, aj1, aj2, aj3, aj4, aj5, aj6)
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])


    ## for MPC
    U = ca.SX.sym('U', 8, N)
    X = ca.SX.sym('X', 17, N+1)
    P = ca.SX.sym('P', 17+17)


    ### define
    X[:, 0] = P[:17] # initial condiction

    #### define the relationship within the horizon
    for i in range(N):
        f_value = f(X[:, i], U[:, i])
        X[:, i+1] = X[:, i] + f_value*T

    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

    Q = np.diag([10.0,10.0,1.0,10.0,10.0,10.0,10.0,10.0,10.0,
                     1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    R = np.diag([0.2,0.2,1.0,1.0,1.0,1.0,0.6,0.5])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        # new type to calculate the matrix times
        obj = obj + (X[:, i]-P[17:]).T @ Q @ (X[:, i]-P[17:]) + U[:, i].T @ R @ U[:, i]

    #### constrains
    g = [] # equal constrains
    for i in range(N+1):
        g.append(X[0, i])
        g.append(X[1, i])

    qp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vcat(g)} # here also can use ca.vcat(g) or ca.vertcat(*g)
    opts_setting = {'print_time':1, 'c1':1e-5, 'hessian_approximation':'limited-memory','qpsol': 'osqp'}
    # opts_setting = {}
    solver = ca.nlpsol('solver', 'sqpmethod', qp_prob, opts_setting)


    # Simulation
    lbg = -10.0
    ubg = 10.0
    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(-10)
        ubx.append(10)
        lbx.append(-10)
        ubx.append(10)
        lbx.append(-10)
        ubx.append(10)
        lbx.append(-10)
        ubx.append(10)
        lbx.append(-10)
        ubx.append(10)
        lbx.append(-10)
        ubx.append(10)
        lbx.append(-10)
        ubx.append(10)
        lbx.append(-10)
        ubx.append(10)

    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
    xs = np.array([1.5, 1.5, 0.0, 1.0, 1.0 , 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1) # final state
    u0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]*N).reshape(-1, 8) # initial control]*N).reshape(-1, 8)# np.ones((N, 2)) # controls
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    ### inital test
    if True:
        # set parameter
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)
        u_sol = ca.reshape(res['x'], 8, N) # one can only have this shape of the output
        ff_value = ff(u_sol, c_p) # [n_states, N+1]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    x_c = np.array(x_c[0])
    u_c = np.array(u_sol)
    plt.plot(x_c[0,:],x_c[1,:])
    plt.subplot(2, 1, 2)
    plt.plot(u_c.transpose())
    plt.show()
