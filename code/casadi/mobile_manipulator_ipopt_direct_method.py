#!/usr/bin/env python
# coding=utf-8

import casadi as ca
import numpy as np
import time

from draw import Draw_MPC_point_stabilization_v1


if __name__ == '__main__':
    T = 0.1
    N = 80
    av_max = 3.0
    wv_max = 3.0

    opti = ca.Opti()
    # control variables, linear acceleration av and angle acceletation aw for mobile base. aj for manipulator joints
    opt_controls = opti.variable(N, 8)
    av = opt_controls[:, 0]
    aw = opt_controls[:, 1]
    aj1 = opt_controls[:, 2]
    aj2 = opt_controls[:, 3]
    aj3 = opt_controls[:, 4]
    aj4 = opt_controls[:, 5]
    aj5 = opt_controls[:, 6]
    aj6 = opt_controls[:, 7]


    opt_states = opti.variable(N+1, 17)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]
    j1 = opt_states[:, 3] 
    j2 = opt_states[:, 4]
    j3 = opt_states[:, 5]
    j4 = opt_states[:, 6]
    j5 = opt_states[:, 7]
    j6 = opt_states[:, 8]
    v = opt_states[:, 9]
    omega = opt_states[:, 10]
    j1_dot = opt_states[:, 11]
    j2_dot = opt_states[:, 12]
    j3_dot = opt_states[:, 13]
    j4_dot = opt_states[:, 14]
    j5_dot = opt_states[:, 15]
    j6_dot = opt_states[:, 16]

    # parameters it's better to read the initial value from ST-rrt
    opt_x0 = opti.parameter(17)
    opt_xs = opti.parameter(17) # the intermidia state may also be the parameter
    # create model
    STATE_DIM = 17
    INPUT_DIM = 8
    BASE_INPUT = 2
    ARM_INPUT = 6
    POSITION_STATE_DIM = 9
    VELOCITY_STATE_DIM = 8

    # system dynamics for mobile manipulator
    f = lambda x_, u_: ca.vertcat(*[
        x_[POSITION_STATE_DIM]*ca.cos(x_[2]), x_[POSITION_STATE_DIM]*ca.sin(x_[2]), x_[POSITION_STATE_DIM+1],
        x_[POSITION_STATE_DIM+2],x_[POSITION_STATE_DIM+3],x_[POSITION_STATE_DIM+4],x_[POSITION_STATE_DIM+5],x_[POSITION_STATE_DIM+6],x_[POSITION_STATE_DIM+7],
        u_[0], u_[1], u_[2], u_[3], u_[4], u_[5], u_[6], u_[7]
        ])

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :]==x_next)

    ## define the cost function
    ### some addition parameters
    Q = np.diag([10.0,10.0,1.0,10.0,10.0,10.0,10.0,10.0,10.0,
                     1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    R = np.diag([0.2,0.2,1.0,1.0,1.0,1.0,0.6,0.5])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :]-opt_xs.T), Q, (opt_states[i, :]-opt_xs.T).T]) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])

    opti.minimize(obj)

    #### boundrary and control conditions
    # opti.subject_to(opti.bounded(-2.0, x, 2.0))
    # opti.subject_to(opti.bounded(-2.0, y, 2.0))
    opti.subject_to(opti.bounded(-3, av, 3))
    opti.subject_to(opti.bounded(-3, aw, 3))
    opti.subject_to(opti.bounded(-2, aj1, 2))
    opti.subject_to(opti.bounded(-2, aj2, 2))
    opti.subject_to(opti.bounded(-2, aj3, 2))
    opti.subject_to(opti.bounded(-2, aj4, 2))
    opti.subject_to(opti.bounded(-2, aj5, 2))
    opti.subject_to(opti.bounded(-2, aj6, 2))


    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':5, 'print_time':1, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
    # opts_setting = {'c1':1e-5, 'hessian_approximation':'limited-memory','qpsol':'osqp'}
    opti.solver('ipopt', opts_setting)
    # opti.solver('sqpmethod',opts_setting)
    final_state = np.array([1.5, 1.5, 0.0, 1.0, 1.0 , 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    opti.set_value(opt_xs, final_state) # may be change to next state

    t0 = 0
    init_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    current_state = init_state.copy()
    next_states = np.zeros((N+1, 17))
    # warm start by yq
    # for i in range(N+1):
    #     tau = i / N
    #     next_states[i, :] = np.array(tau*final_state + (1-tau)*init_state)
    u0 = np.zeros((N, 8))
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []

    ## add by yq

    opti.set_value(opt_x0, current_state)
    opti.set_initial(opt_controls, u0)
    opti.set_initial(opt_states, next_states)
    sol = opti.solve()
    u = sol.value(opt_controls)
    x = sol.value(opt_states)
    # print(u)
    # print(x.shape)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)

    plt.plot(x[:,0],x[:,1])
    np.savetxt("casadi_ipopt_x.txt",x)
    plt.subplot(2, 1, 2)
    plt.plot(u)
    np.savetxt("casadi_ipopt_u.txt",u)
    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=init_state, target_state=final_state, robot_states=x, export_fig=False)
