import numpy as np
import matplotlib.pyplot as plt
from cofig import utils, A, states, observation, n_states, n_observation, n_actions, create_B_matrix,\
    active_inference_with_planning, residual, D_update, change_C


class Agent:

    def __init__(self, starting_state):

        self.init_state = starting_state
        self.current_state = self.init_state
        print(f'Starting state is {starting_state}')

    def step(self, action_label):

        x = self.current_state

        if action_label == "UP":
            next_x = np.where(x < 8, x + 1, x)
        elif action_label == "DOWN":
            next_x = np.where(x > 0, x - 1, x)
        elif action_label == "STAY":
            next_x = x

        self.current_state = (next_x)

        obs = self.current_state

        return obs

    def reset(self):
        self.current_state = self.init_state
        print(f'Re-initialized location to {self.init_state}')
        obs = self.current_state
        print(f'..and sampled observation {obs}')

        return obs


class Subject:
    def __init__(self):
        # Create instances of the three environments
        self.env_r = Agent(5)
        self.env_s = Agent(2)
        self.env_i = Agent(6)
        self.trajectory = []

    # Implement the active inference process
    def run_active_inference(self, epochs):
        # Fill out the components of the generative model for each environment inside this method
        """ Fill out the components of the generative model """
        A_R = A.copy()
        B_R = create_B_matrix()
        C_R = utils.onehot(observation.index(5), n_observation)  # preference
        D_R = utils.onehot(states.index(5), n_states)  # priors over initial state
        actions = ["UP", "DOWN", "STAY"]

        """ Fill out the components of the generative model """
        A_S = A.copy()
        B_S = create_B_matrix()
        # Set the preference of the Symbolic as 4, while its starting state is 0. So it could be thought as a perturbation.
        C_S = utils.onehot(observation.index(5), n_observation)
        D_S = utils.onehot(states.index(0), n_states)
        actions = ["UP", "DOWN", "STAY"]

        """ Fill out the components of the generative model """
        A_I = A.copy()
        B_I = create_B_matrix()
        # We set a sligt perturbation of the Imaginary here
        C_I = utils.onehot(observation.index(6), n_observation)
        D_I = utils.onehot(states.index(6), n_states)
        actions = ["UP", "DOWN", "STAY"]

        for j in range(epochs):
            # Run active inference in each environment using the appropriate generative model
            if j == 0:
                obs_idx_r, qs_curr_r, F_r = active_inference_with_planning(A_R, B_R, C_R, D_R, 5, n_actions, self.env_r,
                                                                           policy_len=2, T=2)
                obs_idx_s, qs_curr_s, F_s = active_inference_with_planning(A_S, B_S, C_S, D_S, 0, n_actions, self.env_s,
                                                                           policy_len=4, T=1)
                obs_idx_i, qs_curr_i, F_i = active_inference_with_planning(A_I, B_I, C_I, D_I, 6, n_actions, self.env_i,
                                                                           policy_len=2, T=1)
            else:
                obs_idx_r, qs_curr_r, F_r = active_inference_with_planning(A_R, B_R, C_R, D_R, obs_idx_r, n_actions,
                                                                           self.env_r, policy_len=2, T=2)
                obs_idx_s, qs_curr_s, F_s = active_inference_with_planning(A_S, B_S, C_S, D_S, obs_idx_s, n_actions,
                                                                           self.env_s, policy_len=4, T=1)
                obs_idx_i, qs_curr_i, F_i = active_inference_with_planning(A_I, B_I, C_I, D_I, obs_idx_i, n_actions,
                                                                           self.env_i, policy_len=2, T=1)
            # Update priors based on residual free energy
            # Residual free energy passage
            R = residual(C_R, qs_curr_r) + residual(C_S, qs_curr_s) + residual(C_I, qs_curr_i)
            D_R = D_update(D_R, F_r, R, 2)
            D_S = D_update(D_S, F_s, R, 0.5)
            D_I = D_update(D_I, F_i, R, 1)

            self.trajectory.append([obs_idx_r, obs_idx_s, obs_idx_i])

        print(self.trajectory)

    # Visulize the dynamics in a 3D space.

    def plot_trajectory(self):
        # Plot the 3D orbit
        trajectory = np.array(self.trajectory)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]
        # Set the axis limits
        ax.set_xlim3d(0, 10)
        ax.set_ylim3d(0, 10)
        ax.set_zlim3d(0, 10)
        # Plot the trajectory as a line
        ax.plot(x, y, z, c='gray', linewidth=2, linestyle='-')

        # Plot the individual points
        ax.scatter(x, y, z, c='gray', s=25, marker='o', )

        # Customize axis labels and title
        ax.set_xlabel('The Real', fontsize=12)
        ax.set_ylabel('The Symbolic', fontsize=12)
        ax.set_zlabel('The Imaginary', fontsize=12)

        # Customize the grid and background
        ax.grid(False)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # Customize the legend
        ax.legend(loc='upper right', fontsize=10)

        plt.show()


class Sync:
    def __init__(self):
        self.env_r = Agent(5)
        self.env_s = Agent(2)
        self.env_i = Agent(6)
        self.trajectory_a = []
        self.trajectory_b = []

    def run_active_inference(self, epochs):
        obs_idx_s_b = 4
        obs_idx_s_a = 2

        A_R_a = A.copy()
        B_R_a = create_B_matrix()
        C_R_a = utils.onehot(observation.index(8), n_observation)  # preference
        D_R_a = utils.onehot(states.index(8), n_states)  # priors over initial state

        A_S_a = A.copy()
        B_S_a = create_B_matrix()
        C_S_a = utils.onehot(observation.index(1), n_observation)
        D_S_a = utils.onehot(states.index(1), n_states)

        A_I_a = A.copy()
        B_I_a = create_B_matrix()
        C_I_a = utils.onehot(observation.index(6), n_observation)
        D_I_a = utils.onehot(states.index(6), n_states)

        A_R_b = A.copy()
        B_R_b = create_B_matrix()
        C_R_b = utils.onehot(observation.index(3), n_observation)  # preference
        D_R_b = utils.onehot(states.index(3), n_states)  # priors over initial state

        A_S_b = A.copy()
        B_S_b = create_B_matrix()
        C_S_b = utils.onehot(observation.index(6), n_observation)
        D_S_b = utils.onehot(states.index(6), n_states)

        A_I_b = A.copy()
        B_I_b = create_B_matrix()
        C_I_b = utils.onehot(observation.index(8), n_observation)
        D_I_b = utils.onehot(states.index(8), n_states)

        for j in range(epochs):
            # Run active inference in each environment using the recurrent generative model
            if j == 0:
                obs_idx_r_a, qs_curr_r_a, F_r_a = active_inference_with_planning(
                    A_R_a, B_R_a, C_R_a, D_R_a, 8, n_actions, self.env_r, policy_len=2, T=2)
                C_S_a = change_C(obs_idx_s_b)
                obs_idx_s_a, qs_curr_s_a, F_s_a = active_inference_with_planning(
                    A_S_a, B_S_a, C_S_a, D_S_a, 1, n_actions, self.env_s, policy_len=2, T=1)
                obs_idx_i_a, qs_curr_i_a, F_i_a = active_inference_with_planning(
                    A_I_a, B_I_a, C_I_a, D_I_a, 6, n_actions, self.env_i, policy_len=2, T=1)
            else:
                obs_idx_r_a, qs_curr_r_a, F_r_a = active_inference_with_planning(
                    A_R_a, B_R_a, C_R_a, D_R_a, obs_idx_r_a, n_actions, self.env_r, policy_len=2, T=2)
                C_S_a = change_C(obs_idx_s_b)
                obs_idx_s_a, qs_curr_s_a, F_s_a = active_inference_with_planning(
                    A_S_a, B_S_a, C_S_a, D_S_a, obs_idx_s_a, n_actions, self.env_s, policy_len=2, T=1)
                obs_idx_i_a, qs_curr_i_a, F_i_a = active_inference_with_planning(
                    A_I_a, B_I_a, C_I_a, D_I_a, obs_idx_i_a, n_actions, self.env_i, policy_len=2, T=1)
                # Update priors based on residual free energy
            # Residual free energy passage
            R_a = residual(C_R_a, qs_curr_r_a) + residual(C_S_a, qs_curr_s_a) + residual(C_I_a, qs_curr_i_a)
            D_R_a = D_update(D_R_a, F_r_a, R_a, 2)
            D_S_a = D_update(D_S_a, F_s_a, R_a, 0.5)
            D_I_a = D_update(D_I_a, F_i_a, R_a, 1)

            self.trajectory_a.append([obs_idx_r_a, obs_idx_s_a, obs_idx_i_a])

            if j == 0:
                obs_idx_r_b, qs_curr_r_b, F_r_b = active_inference_with_planning(
                    A_R_b, B_R_b, C_R_b, D_R_b, 3, n_actions, self.env_r, policy_len=2, T=2)
                C_S_b = change_C(obs_idx_s_a)
                obs_idx_s_b, qs_curr_s_b, F_s_b = active_inference_with_planning(
                    A_S_b, B_S_b, C_S_b, D_S_b, 6, n_actions, self.env_s, policy_len=4, T=1)
                obs_idx_i_b, qs_curr_i_b, F_i_b = active_inference_with_planning(
                    A_I_b, B_I_b, C_I_b, D_I_b, 8, n_actions, self.env_i, policy_len=2, T=1)
            else:
                obs_idx_r_b, qs_curr_r_b, F_r_b = active_inference_with_planning(
                    A_R_b, B_R_b, C_R_b, D_R_b, obs_idx_r_b, n_actions, self.env_r, policy_len=2, T=2)
                obs_idx_s_b, qs_curr_s_b, F_s_b = active_inference_with_planning(
                    A_S_b, B_S_b, C_S_b, D_S_b, obs_idx_s_b, n_actions, self.env_s, policy_len=4, T=1)
                C_S_b = change_C(obs_idx_s_a)
                obs_idx_i_b, qs_curr_i_b, F_i_b = active_inference_with_planning(
                    A_I_b, B_I_b, C_I_b, D_I_b, obs_idx_i_b, n_actions, self.env_i, policy_len=2, T=1)
            R_b = residual(C_R_b, qs_curr_r_b) + residual(C_S_b, qs_curr_s_b) + residual(C_I_b, qs_curr_i_b)
            D_R_b = D_update(D_R_b, F_r_b, R_b, 0.5)
            D_S_b = D_update(D_S_b, F_s_b, R_b, 2)
            D_I_b = D_update(D_I_b, F_i_b, R_b, 2)

            self.trajectory_b.append([obs_idx_r_b, obs_idx_s_b, obs_idx_i_b])

    def plot_trajectories(self):
        # Plot the trajectories of both agents
        trajectory_a = np.array(self.trajectory_a)
        trajectory_b = np.array(self.trajectory_b)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Agent A trajectory
        x_a = trajectory_a[:, 0]
        y_a = trajectory_a[:, 1]
        z_a = trajectory_a[:, 2]
        ax.plot(x_a, y_a, z_a, c='red', linewidth=1, linestyle='-', label='Agent A')

        # Agent B trajectory
        x_b = trajectory_b[:, 0]
        y_b = trajectory_b[:, 1]
        z_b = trajectory_b[:, 2]
        ax.plot(x_b, y_b, z_b, c='black', linewidth=1, linestyle='-', label='Agent B')

        # Customize axis labels and title
        ax.set_xlim3d(0, 8)
        ax.set_ylim3d(0, 8)
        ax.set_zlim3d(0, 8)

        # Plot the individual points
        ax.scatter(x_a, y_a, z_a, c='red', s=25, marker='o')
        ax.scatter(x_b, y_b, z_b, c='black', s=25, marker='o')

        # Customize axis labels and title
        ax.set_xlabel('The Real', fontsize=12)
        ax.set_ylabel('The Symbolic', fontsize=12)
        ax.set_zlabel('The Imaginary', fontsize=12)
        ax.set_title('3D Orbit of Observations', fontsize=16)

        # Customize the grid and background
        ax.grid(False)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # Customize the legend
        ax.legend(loc='upper right', fontsize=10)

        plt.show()




