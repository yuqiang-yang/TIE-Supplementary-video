#!/usr/bin/env python
# coding=utf-8

import casadi as ca
import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scipy.interpolate import interp1d

class Mpc_controller:
    def __init__(self, global_planed_traj, N = 10, desired_v = 1.0, v_max = 2.0, w_max = 2.0, ref_gap = 2):
        self.N, self.desired_v, self.ref_gap, self.T = N, desired_v, ref_gap, 0.1
        
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj_len = N // ref_gap + 1

        # setup mpc problem
        opti = ca.Opti()
        opt_controls = opti.variable(N, 2)
        v, w = opt_controls[:, 0], opt_controls[:, 1]

        opt_states = opti.variable(N+1, 3)
        x, y, theta = opt_states[:, 0], opt_states[:, 1], opt_states[:, 2]

        # parameters 
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3 * self.ref_traj_len) # the intermidia state may also be the parameter

        # system dynamics for mobile manipulator
        f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])

        # init_condition
        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(N):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*self.T
            opti.subject_to(opt_states[i+1, :]==x_next)

        # define the cost function
        Q = np.diag([10.0,10.0,0.0])
        R = np.diag([0.02,0.02])
        obj = 0 
        for i in range(N):
            obj = obj +ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
            if i % ref_gap == 0:
                nn = i // ref_gap
                obj = obj + ca.mtimes([(opt_states[i, :]-opt_xs[nn*3:nn*3+3].T), Q, (opt_states[i, :]-opt_xs[nn*3:nn*3+3].T).T])

        opti.minimize(obj)

        # boundrary and control conditions
        opti.subject_to(opti.bounded(-v_max, v, v_max))
        opti.subject_to(opti.bounded(-w_max, w, w_max))
        
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
        opti.solver('ipopt', opts_setting)
        # opts_setting = { 'qpsol':'osqp','hessian_approximation':'limited-memory','max_iter':200,'convexify_strategy':'regularize','beta':0.5,'c1':1e-4,'tol_du':1e-3,'tol_pr':1e-6}
        # opti.solver('sqpmethod',opts_setting)
        
        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_controls = opt_controls
        self.opt_states = opt_states
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
    def make_ref_denser(self, ref_traj, ratio = 50):
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(0, len(ref_traj) - 1, num=len(ref_traj) * ratio)

        interp_func_x = interp1d(x_orig, ref_traj[:, 0], kind='linear')
        interp_func_y = interp1d(x_orig, ref_traj[:, 1], kind='linear')

        uniform_x = interp_func_x(new_x)
        uniform_y = interp_func_y(new_x)
        ref_traj = np.stack((uniform_x, uniform_y), axis=1)
        
        return ref_traj
    
    def solve(self, x0):
        ref_traj = self.find_reference_traj(x0, self.ref_traj)
        # fake a yaw angle
        ref_traj = np.concatenate((ref_traj, np.zeros((ref_traj.shape[0], 1))), axis=1).reshape(-1, 1)
        
        self.opti.set_value(self.opt_xs, ref_traj.reshape(-1, 1)) 
        u0 = np.zeros((self.N, 2)) if self.last_opt_u_controls is None else self.last_opt_u_controls
        x0 = np.zeros((self.N+1, 3)) if self.last_opt_x_states is None else self.last_opt_x_states

        self.opti.set_value(self.opt_x0, x_history[-1])
        self.opti.set_initial(self.opt_controls, u0)
        self.opti.set_initial(self.opt_states, x0)

        sol = self.opti.solve()

        self.last_opt_u_controls = sol.value(self.opt_controls)
        self.last_opt_x_states = sol.value(self.opt_states)

        return self.last_opt_u_controls, self.last_opt_x_states
    def reset(self):
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        
    def find_reference_traj(self, x0, global_planed_traj):
        ref_traj_pts = []
        # find the nearest point in global_planed_traj
        nearest_idx = np.argmin(np.linalg.norm(global_planed_traj - x0[:2].reshape((1, 2)), axis=1))
        desire_arc_length = self.desired_v * self.ref_gap * self.T 
        cum_dist = np.cumsum(np.linalg.norm(np.diff(global_planed_traj, axis=0), axis=1))

        # select the reference points from the nearest point to the end of global_planed_traj
        for i in range(nearest_idx, len(global_planed_traj) - 1):
            if cum_dist[i] - cum_dist[nearest_idx] >= desire_arc_length * len(ref_traj_pts):
                ref_traj_pts.append(global_planed_traj[i, :])
                if len(ref_traj_pts) == self.ref_traj_len:
                    break
        # if the target is reached before the reference trajectory is complete, add the last point of global_planed_traj 
        while len(ref_traj_pts) < self.ref_traj_len:
            ref_traj_pts.append(global_planed_traj[-1, :])
        return np.array(ref_traj_pts)
        
        
if __name__ == '__main__':

    global_planed_traj = np.load("gt_exe_path0/gt_exe_path3")[:, :2]
    mpc = Mpc_controller(global_planed_traj)
    
    t0 = 0
    init_state = np.array([global_planed_traj[0, 0], global_planed_traj[0, 1], 1.57])
    
    x_history = [init_state]
    u_history = []
    
    # Receding horizon control
    for i in range(50):
        opt_u_controls, opt_x_states = mpc.solve(x_history[-1])
        x_history.append(opt_x_states[2, :])
        u_history.append(opt_u_controls[2, :])
    
    x_history = np.array(x_history)
    u_history = np.array(u_history)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(x_history[:,0],x_history[:,1], label='local mpc')
    plt.plot(global_planed_traj[:, 0], global_planed_traj[:, 1], '--', label='Global Ref')
    plt.subplot(2, 1, 2)
    plt.plot(u_history[:,0])
    plt.plot(u_history[:,1])
    plt.legend()
    plt.show()
    
