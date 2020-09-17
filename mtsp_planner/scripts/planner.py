#!/usr/bin/env python2
"""
Custom TSP planner
@author: R.Penicka, T.Baca
"""

import rospy
import os
import copy
import itertools
import numpy as np
import dubins
this_script_path = os.path.dirname(__file__)

# MRS ROS messages
from mtsp_msgs.msg import TspProblem
from mrs_msgs.msg import TrajectoryReference

# the TSP problem class
from mtsp_problem_loader.tsp_problem import *

from solvers.tsp_solvers import *
from solvers import toppra_trajectory
import solvers.tsp_trajectory

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import argparse

# TODO: optimize topp-ra
# TODO: test on different scenarios

use_toppra = False

class TspPlanner:

    def __init__(self, use_ros=True):

        if use_ros:
            print("runing with ros")

            rospy.init_node('tsp_planner', anonymous=True)

            self._max_velocity = rospy.get_param('~max_velocity', 4)
            self._max_acceleration = rospy.get_param('~max_acceleration', 2)
            self._turning_velocity = rospy.get_param('~turning_velocity', 2)
            self._plot = rospy.get_param('~plot', False)

    
            print("using max_velocity", self._max_velocity)
            print("using max_acceleration", self._max_acceleration)
            print("using plot", self._plot)
            print("using turning_velocity", self._turning_velocity)
    
            # based on the velocity and acceleration
            self.turning_radius = (self._max_velocity * self._max_velocity) / self._max_acceleration
    
            # initiate ROS publishers
            self.publisher_trajectory_1 = rospy.Publisher("~trajectory_1_out", TrajectoryReference, queue_size=1)
            self.publisher_trajectory_2 = rospy.Publisher("~trajectory_2_out", TrajectoryReference, queue_size=1)

            rate = rospy.Rate(1.0)
            rate.sleep()
    
            # initiate ROS subscribers
            self.subscriber_problem = rospy.Subscriber("~problem_in", TspProblem, self.callbackProblem, queue_size=1)
    
            rospy.loginfo('Planner initialized')

            rospy.spin()
            
        else:
            print("runing as bare script")
            config_file = os.path.join(this_script_path, "../config/simulation.yaml")
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
                print("using default params from config file", config_file, ":", loaded_config)
            
            arg_parser = argparse.ArgumentParser(description='MTSP planner')
            arg_parser.add_argument('--max_velocity', type=float, default=loaded_config['max_velocity'])
            arg_parser.add_argument('--max_acceleration', type=float, default=loaded_config['max_acceleration'])
            arg_parser.add_argument('--turning_velocity', type=float, default=loaded_config['turning_velocity'])
            arg_parser.add_argument('--plot', type=bool, default=loaded_config['plot'])
            arg_parser.add_argument('--problem', type=str, default=os.path.join(this_script_path, loaded_config['problem']))
            
            args = arg_parser.parse_args()
            print("running with args", args)
            
            self._max_velocity = args.max_velocity
            self._max_acceleration = args.max_acceleration
            self._plot = True
            self._turning_velocity = args.turning_velocity
            
            print("using max_velocity", self._max_velocity)
            print("using max_acceleration", self._max_acceleration)
            print("using plot", self._plot)
            print("using turning_velocity", self._turning_velocity)
            
            # self._turning_radius = (self._max_velocity*self._max_velocity)/self._max_acceleration
            
            # load_problem
            tsp_problem = MTSPProblem.load_problem(args.problem) 
            
            self.plan_trajectory(tsp_problem)

    def callbackProblem(self, problem):
        """ros callback for mtsp problem message"""
        rospy.loginfo_throttle(1.0, 'Received the TSP problem')

        # copy the points
        targets = []
        starts = []
        for point in problem.points:
            target = [point.idx, point.x, point.y]
            targets.append(target)
        for point in problem.start_points:
            start = [point.idx, point.x, point.y]
            starts.append(start)

        # create the python tsp_problem object
        tsp_problem = MTSPProblem(problem.name, problem.type, problem.comment, problem.dimension, problem.number_of_robots, starts, problem.neighborhood_radius, problem.edge_weight_type, targets)

        rospy.loginfo('The problem is {}'.format(problem))

        trajectories_samples = self.plan_trajectory(tsp_problem)

        # # | ------------------ plot velocity profile ----------------- |
        # plt.show()

        # # | --------------- create the ROS trajectories -------------- |
        trajectory = solvers.tsp_trajectory.TSPTrajectory(self._max_velocity, self._max_acceleration)
        ros_trajectory_1 = trajectory.create_ros_trajectory(trajectories_samples[0], problem.height)
        ros_trajectory_2 = trajectory.create_ros_trajectory(trajectories_samples[1], problem.height)

        # # | ---------------- publish the trajectories ---------------- |
        self.publisher_trajectory_1.publish(ros_trajectory_1)
        self.publisher_trajectory_2.publish(ros_trajectory_2)

        rospy.loginfo('trajectories were published')

    def plan_trajectory(self, tsp_problem): 
        """method for planning the M(D)TSP(N) plans based on tsp_problem"""
        
        tsp_solver = TSPSolver()

        ############### TARGET LOCATIONS CLUSTERING BEGIN ###############
        clusters, cluster_centers = tsp_solver.cluster_kmeans(tsp_problem.targets, tsp_problem.number_of_robots)
        if (dist_euclidean(tsp_problem.start_positions[0][0:2], cluster_centers[0][0:2]) <
            dist_euclidean(tsp_problem.start_positions[0][0:2], cluster_centers[1][0:2])):
            clusters[0].insert(0, tsp_problem.start_positions[0])
            clusters[1].insert(0, tsp_problem.start_positions[1])
        else:
            clusters[0].insert(0, tsp_problem.start_positions[1])
            clusters[1].insert(0, tsp_problem.start_positions[0])

        # clusters, cluster_centers = tsp_solver.cluster_from_start(tsp_problem.targets, tsp_problem.start_positions)

        ############### TARGET LOCATIONS CLUSTERING END ###############

        best_trajectory_samples, best_trajectory_time = self.attempt_plan_trajectory(tsp_problem, tsp_solver, clusters)
        num_collision = check_collisions(best_trajectory_samples)
        initial_trajectory_time = best_trajectory_time + 10 * num_collision


        # TODO: check for collisions
        # TODO: reassign starting points

        while(abs(len(clusters[0]) - len(clusters[1])) > 1):
            clusters = move_points_between_clusters(clusters, cluster_centers)
            trajectory_samples, trajectory_time = self.attempt_plan_trajectory(tsp_problem, tsp_solver, clusters)
            num_collision = check_collisions(trajectory_samples)
            trajectory_time = trajectory_time + 10 * num_collision

            if trajectory_time < best_trajectory_time:
                best_trajectory_time = trajectory_time
                best_trajectory_samples = copy.deepcopy(trajectory_samples)

        print("Initial trajectory time", initial_trajectory_time)
        print("Best trajectory time", best_trajectory_time)

        return best_trajectory_samples


    def attempt_plan_trajectory(self, tsp_problem, tsp_solver, clusters):
        # copy the points
        if self._plot:
            ax = MTSPProblem.plot_problem(tsp_problem, show=False)
            arena_corners = read_world_file_arena(os.path.join(this_script_path, "../../mtsp_state_machine/config/world.yaml"))
            plt.plot([arena_corners[i][0] for i in range(-1, len(arena_corners))],
                     [arena_corners[i][1] for i in range(-1, len(arena_corners))], 'c-', label='fly area')

        # # | -------------------- plot the clusters ------------------- |
        if self._plot:  # plot the clusters
            colors = cm.rainbow(np.linspace(0, 1, tsp_problem.number_of_robots))
            for i in range(tsp_problem.number_of_robots):
                # plt.plot([cluster_centers[i][0]],[cluster_centers[i][1]],'*',color=colors[i])
                plt.plot([c[1] for c in clusters[i]], [c[2] for c in clusters[i]], '.', color=colors[i])

        # # | ---------------------- solve the TSP --------------------- |
        robot_sequences = []
        # for i in range(1):
        for i in range(tsp_problem.number_of_robots):

            ############### TSP SOLVERS PART BEGIN ###############
            # path = tsp_solver.plan_tour_etsp(clusters[i],0) #find decoupled ETSP tour over clusters
            # path = tsp_solver.plan_tour_etspn_decoupled(clusters[i], 0, tsp_problem.neighborhood_radius * 0.8)  # find decoupled ETSPN tour over clusters

            turning_radius = (self._turning_velocity * self._turning_velocity) / self._max_acceleration
            sampler = tsp_trajectory.TSPTrajectory(self._max_velocity, self._max_acceleration)
            path = tsp_solver.plan_tour_dtspn_decoupled(clusters[i], 0, tsp_problem.neighborhood_radius * 0.65, turning_radius)  # find decoupled DTSPN tour over clusters
            # path = tsp_solver.plan_tour_dtspn_noon_bean(clusters[i], 0, tsp_problem.neighborhood_radius * 0.65, turning_radius,
            #                                             turning_velocity=self._turning_velocity, sampler=sampler) # find noon-bean DTSPN tour over clusters

            ############### TSP SOLVERS PART END ###############

            print("path", path)
            robot_sequences.append(path)

            # # | -------------------- plot the solution ------------------- |
            if self._plot:  # plot tsp solution
                sampled_path_all = []
                for pid in range(1, len(path)):
                    if len(path[pid]) == 2 :
                        sampled_path_all += path
                        sampled_path_all += [path[0]]
                        # plt.plot([path[pid - 1][0] , path[pid][0]], [path[pid - 1][1] , path[pid][1]], '-', color=colors[i], lw=0.8, label='trajectory %d' % (i + 1))
                    elif len(path[pid]) == 3 :
                        dubins_path = dubins.shortest_path(path[pid - 1], path[pid], turning_radius)
                        sampled_path , _ = dubins_path.sample_many(0.1)
                        sampled_path_all += sampled_path
                if (len(path[0]) == 3):
                    plt.quiver([p[0] for p in path], [p[1] for p in path], [1] * len(path), [1] * len(path),
                               angles=np.degrees([p[2] for p in path]), width=0.002, headwidth=3, )
                plt.plot([p[0] for p in sampled_path_all] , [p[1] for p in sampled_path_all] , '-', color=colors[i], lw=1.2, label='trajectory %d' % (i + 1))
                plt.plot([p[0] for p in path], [p[1] for p in path], ls='none', marker='x', ms=7, mfc='k', mec='k')

        # Initialize trajectory sampler.
        if use_toppra:
            trajectory = toppra_trajectory.ToppraTrajectory(self._max_velocity, self._max_acceleration)
        else:
            trajectory = tsp_trajectory.TSPTrajectory(self._max_velocity, self._max_acceleration)

        # # | ------------------- sample trajectories ------------------ |
        trajectories_samples = []
        max_trajectory_time = 0
        for i in range(len(robot_sequences)):

            if len(robot_sequences[i][0]) == 2 :
                if use_toppra:
                    single_trajectory_samples, trajectory_time = trajectory.generate_toppra_trajectory(robot_sequences[i])
                else:
                    single_trajectory_samples, trajectory_time = trajectory.sample_trajectory_euclidean(robot_sequences[i])
            elif len(robot_sequences[i][0]) == 3:
                if use_toppra:
                    single_trajectory_samples, trajectory_time = trajectory.generate_toppra_trajectory(robot_sequences[i])
                else:
                    single_trajectory_samples, trajectory_time = trajectory.sample_trajectory_dubins(robot_sequences[i], turning_velocity=self._turning_velocity)

            print("trajectory_time", i+1, "is", trajectory_time)

            # Fix the heading at each point of the trajectory-
            for j in range(len(single_trajectory_samples)):
                sample = single_trajectory_samples[j]
                single_trajectory_samples[j] = (sample[0], sample[1], 0)

            trajectories_samples.append(single_trajectory_samples)
            
            if trajectory_time > max_trajectory_time:
                max_trajectory_time = trajectory_time
            
            if self._plot:  # plot trajectory samples
                plt.plot([p[0] for p in single_trajectory_samples], [p[1] for p in single_trajectory_samples], 'o', markerfacecolor=colors[i], markeredgecolor='k', ms=2.0, markeredgewidth=0.3 , label='samples %d' % (i + 1))
                
        if self._plot:  # add legend to trajectory plot
            plt.legend(loc='upper right')
            plt.title("Time: {:.3f} s".format(max_trajectory_time))
            
        print("maximal time of trajectory is", max_trajectory_time)

        # # | --------------- plot velocity profiles --------------- |
        # if self._plot:  # plot velocity profile
        #     for i in range(len(trajectories_samples)):
        #         trajectory.plot_velocity_profile(trajectories_samples[i], color=colors[i],title = 'Velocity profile %d' % (i + 1))
        
        # # | ----------------------- show plots ---------------------- |
        if self._plot:
            plt.show()
        return trajectories_samples, max_trajectory_time
            
def move_points_between_clusters(clusters, cluster_centres):
    smaller = 0 if len(clusters[0]) < len(clusters[1]) else 1
    bigger = 1 if smaller == 0 else 0

    min_distance = 10000
    idx = -1
    for point in clusters[bigger]:
        idx += 1
        if idx == 0:
            continue
        dist = dist_euclidean(cluster_centres[smaller], point[1:3])
        if dist < min_distance:
            min_distance = dist
            min_index = idx

    clusters[smaller].append(clusters[bigger].pop(min_index))

    return clusters

def check_collisions(trajectory_samples):
    dist_warn = 1
    deadzone = 2
    num_collisons = 0
    time_sample = 0.2

    shorter_trajectory_idx = 0 if len(trajectory_samples[0]) < len(trajectory_samples[1]) else 1
    shorter_trajectory_end = trajectory_samples[shorter_trajectory_idx][-1]
    t = 0
    last_collision = 0
    for point1, point2 in itertools.izip_longest(trajectory_samples[0], trajectory_samples[1], fillvalue=shorter_trajectory_end):
        if dist_euclidean(point1[0:2], point2[0:2]) < dist_warn:
            if (t - last_collision) * time_sample > deadzone:
                num_collisons += 1
                last_collision = t
                print("!!! COLLISION DETECTED Total: {}!!!".format(num_collisons))
        t += 1

    return num_collisons


if __name__ == '__main__':
    myargv = rospy.myargv(argv=sys.argv)
    if "--ros" in myargv:
        try:
            tsp_planner = TspPlanner()
        except rospy.ROSInterruptException:
            pass
    else:
        tsp_planner = TspPlanner(use_ros=False)
        
