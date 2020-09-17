#!/usr/bin/env python
# -*- coding: utf-8 -*-

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def dist_euclidean_squared(coord1, coord2):
    """ euclidean distance between coord1 and coord2"""
    (x1, y1) = coord1
    (x2, y2) = coord2
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

def dist_euclidean(coord1, coord2):
    """ euclidean distance between coord1 and coord2"""
    return math.sqrt(dist_euclidean_squared(coord1, coord2))

class ToppraTrajectory():
    def __init__(self, max_velocity, max_acceleration):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.time_sample = 0.2

    def generate_toppra_trajectory(self, input_path):
        waypoints = np.array([[p[0], p[1]] for p in input_path])

        path = ta.SplineInterpolator(np.linspace(0, 1, len(waypoints)), waypoints)

        # Create velocity bounds, then velocity constraint object
        max_vel = self.max_velocity / math.sqrt(2)
        max_acc = self.max_acceleration / math.sqrt(2)
        vlim_low = np.array([-max_vel, -max_vel])
        vlim_high = np.array([max_vel, max_vel])
        vlim = np.vstack((vlim_low, vlim_high)).T
        # Create acceleration bounds, then acceleration constraint object
        alim_low = np.array([-max_acc, -max_acc])
        alim_high = np.array([max_acc, max_acc])
        alim = np.vstack((alim_low, alim_high)).T
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(
            alim, discretization_scheme=constraint.DiscretizationType.Interpolation)

        # Setup a parametrization instance
        instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')

        # Retime the trajectory, only this step is necessary.
        t0 = time.time()
        jnt_traj, aux_traj = instance.compute_trajectory(0, 0)
        # print("Parameterization time: {:} secs".format(time.time() - t0))
        # Sampling frequency is required to get the time samples correctly.
        # The number of points in self.time_sample is duration*frequency.
        sample_freq = 1 / self.time_sample
        ts_sample = np.linspace(0, jnt_traj.get_duration(),
                                int(jnt_traj.get_duration() * sample_freq))
        # Sampling. This returns a matrix for all DOFs. Accessing specific one is
        # simple: qs_sample[:, 0]
        qs_sample = jnt_traj.eval(ts_sample)
        qds_sample = jnt_traj.evald(ts_sample)
        qdds_sample = jnt_traj.evaldd(ts_sample)

        qds_abs = np.linalg.norm(qds_sample, axis=1)
        qdds_abs = np.linalg.norm(qdds_sample, axis=1)

        # plt.plot(ts_sample, qs_sample)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Joint position (m)")
        # plt.show()

        # figsize = (8, 5)
        # fig = plt.figure(num=4, figsize=figsize)
        # # plt.plot(ts_sample, qds_sample)
        # plt.plot(ts_sample, qds_abs)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Joint velocity (m/s)")
        # plt.show()
        #
        # figsize = (8, 5)
        # fig = plt.figure(num=5, figsize=figsize)
        # # plt.plot(ts_sample, qdds_sample)
        # plt.plot(ts_sample, qdds_abs)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Joint acceleration (m/s^2)")
        # plt.show()

        return qs_sample.tolist(), max(ts_sample)

    def sample_toppra_trjaceotry(self):
        pass

    def create_ros_trajectory(self):
        pass

    def plot_velocity_profile(self, samples, color='k', title='Velocity profile'):

        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_title(title)
        ax.set_ylabel('velocity [m/s]')

        ax.set_xlabel('time [s]')

        velocities = [0]
        for i in range(1, len(samples)):
            dist = dist_euclidean(samples[i - 1][0:2], samples[i][0:2])
            velocities.append(dist / self.time_sample)
        velocities_time = [self.time_sample * i for i in range(len(velocities))]

        accelerations = [0]
        for i in range(1, len(velocities)):
            vel_change = velocities[i] - velocities[i - 1]
            accelerations.append(vel_change / self.time_sample)

        accelerations_time = [self.time_sample * i for i in range(len(accelerations))]

        plt.axhline(self.max_velocity, 0, len(velocities), ls='-', color='k')
        plt.plot(velocities_time, velocities, '-', color=color, label='velocity')
        plt.axhline(self.max_acceleration, 0, len(accelerations), ls='-.', color='k')
        plt.axhline(-self.max_acceleration, 0, len(accelerations), ls='-.', color='k')
        plt.plot(accelerations_time, accelerations, '-.', color=color, label='acc')
        ax.legend(loc='upper right')
        ax2 = ax.twinx()
        ax2.set_ylabel('acceleration [m/s^2]')
