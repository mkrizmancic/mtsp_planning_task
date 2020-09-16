#!/usr/bin/env python
# -*- coding: utf-8 -*-

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time

class ToppraTrajectory():
    def __init__(self, max_velocity, max_acceleration):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def generate_toppra_trajectory(self, input_path):
        waypoints = np.array([[p[0], p[1]] for p in input_path])

        path = ta.SplineInterpolator(np.linspace(0, 1, len(waypoints)), waypoints)

        # Create velocity bounds, then velocity constraint object
        vlim_low = np.array([-self.max_velocity, -self.max_velocity])
        vlim_high = np.array([self.max_velocity, self.max_velocity])
        vlim = np.vstack((vlim_low, vlim_high)).T
        # Create acceleration bounds, then acceleration constraint object
        alim_low = np.array([-self.max_acceleration, -self.max_acceleration])
        alim_high = np.array([self.max_acceleration, self.max_acceleration])
        alim = np.vstack((alim_low, alim_high)).T
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(
            alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

        # Setup a parametrization instance
        instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')

        # Retime the trajectory, only this step is necessary.
        t0 = time.time()
        jnt_traj, aux_traj = instance.compute_trajectory(0, 0)
        print("Parameterization time: {:} secs".format(time.time() - t0))
        # Sampling frequency is required to get the time samples correctly.
        # The number of points in ts_sample is duration*frequency.
        sample_freq = 50
        ts_sample = np.linspace(0, jnt_traj.get_duration(),
                                int(jnt_traj.get_duration() * sample_freq))
        # Sampling. This returns a matrix for all DOFs. Accessing specific one is
        # simple: qs_sample[:, 0]
        qs_sample = jnt_traj.eval(ts_sample)
        qds_sample = jnt_traj.evald(ts_sample)
        qdds_sample = jnt_traj.evaldd(ts_sample)

        # plt.plot(ts_sample, qs_sample)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Joint position (m)")
        # plt.show()
        #
        # plt.plot(ts_sample, qds_sample)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Joint velocity (m/s)")
        # plt.show()
        #
        # plt.plot(ts_sample, qdds_sample)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Joint acceleration (m/s^2)")
        # plt.show()

        return qs_sample.tolist(), max(ts_sample)

    def sample_toppra_trjaceotry(self):
        pass

    def create_ros_trajectory(self):
        pass
