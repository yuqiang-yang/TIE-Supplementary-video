#!/usr/bin/env python
# coding=UTF-8


import numpy as np
import casadi as ca
from acados_template import AcadosModel

class MobileManipulatorModel(object):
    def __init__(self,):
        model = AcadosModel() #  ca.types.SimpleNamespace()
        constraint = ca.types.SimpleNamespace()
        # control inputs
        av = ca.SX.sym('av')
        aw = ca.SX.sym('aw')
        aj1 = ca.SX.sym('aj1')
        aj2 = ca.SX.sym('aj2')
        aj3 = ca.SX.sym('aj3')
        aj4 = ca.SX.sym('aj4')
        aj5 = ca.SX.sym('aj5')
        aj6 = ca.SX.sym('aj6')
        controls = ca.vertcat(av, aw, aj1, aj2, aj3, aj4, aj5, aj6)
        n_controls = controls.size()[0]
        # model states
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
        w = ca.SX.sym('w')
        j1_dot = ca.SX.sym('j1_dot')
        j2_dot = ca.SX.sym('j2_dot')
        j3_dot = ca.SX.sym('j3_dot')
        j4_dot = ca.SX.sym('j4_dot')
        j5_dot = ca.SX.sym('j5_dot')
        j6_dot = ca.SX.sym('j6_dot')
        states = ca.vertcat(x, y, theta, j1, j2, j3, j4, j5, j6, v, w, j1_dot, j2_dot, j3_dot, j4_dot, j5_dot, j6_dot)


        rhs = [v*ca.cos(theta), v*ca.sin(theta), w, j1_dot, j2_dot, j3_dot, j4_dot, j5_dot, j6_dot,
               av, aw, aj1, aj2, aj3, aj4, aj5, aj6]
        
        # function
        f = ca.Function('f', [states, controls], [ca.vcat(rhs)], ['state', 'control_input'], ['rhs'])
        # acados model
        x_dot = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = 'mobile_manipulator'

        # constraint
        constraint.x_min = -99
        constraint.x_max = 99
        constraint.y_min = -99
        constraint.y_max = 99
        constraint.theta_min = -np.pi
        constraint.theta_max = np.pi
        constraint.j1_min = -2*np.pi
        constraint.j1_max = 2*np.pi
        constraint.j2_min = -2*np.pi
        constraint.j2_max = 2*np.pi
        constraint.j3_min = -2*np.pi
        constraint.j3_max = 2*np.pi
        constraint.j4_min = -2*np.pi
        constraint.j4_max = 2*np.pi
        constraint.j5_min = -2*np.pi
        constraint.j5_max = 2*np.pi
        constraint.j6_min = -2*np.pi
        constraint.j6_max = 2*np.pi

        constraint.v_max = 10.0
        constraint.v_min = -10.0
        constraint.w_max = 10.0
        constraint.w_min = -10.0
        constraint.j1_dot_max = 10.0
        constraint.j1_dot_min = -10.0
        constraint.j2_dot_max = 10.0
        constraint.j2_dot_min = -10.0
        constraint.j3_dot_max = 10.0
        constraint.j3_dot_min = -10.0
        constraint.j4_dot_max = 10.0
        constraint.j4_dot_min = -10.0
        constraint.j5_dot_max = 10.0
        constraint.j5_dot_min = -10.0
        constraint.j6_dot_max = 10.0
        constraint.j6_dot_min = -10.0

        constraint.av_min = -3.0
        constraint.av_max = 3.0
        constraint.aw_min = -3.0
        constraint.aw_max = 3.0
        constraint.aj1_min = -2.0
        constraint.aj1_max = 2.0
        constraint.aj2_min = -2.0
        constraint.aj2_max = 2.0
        constraint.aj3_min = -2.0
        constraint.aj3_max = 2.0
        constraint.aj4_min = -2.0
        constraint.aj4_max = 2.0
        constraint.aj5_min = -2.0
        constraint.aj5_max = 2.0
        constraint.aj6_min = -2.0
        constraint.aj6_max = 2.0

        constraint.expr = ca.vcat([av, aw, aj1, aj2, aj3, aj4, aj5, aj6])

        self.model = model
        self.constraint = constraint