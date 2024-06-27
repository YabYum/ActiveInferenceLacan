import numpy as np
import matplotlib.pyplot as plt
import itertools
from pymdp import utils

states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
observation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
actions = ["UP", "DOWN", "STAY"]
n_states = 11
n_observation = 11
n_actions = 3

# Create likelihood matrix A
A = np.array([
    [0.85, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.05, 0.80, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00],
    [0.04, 0.05, 0.76, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.00, 0.00],
    [0.03, 0.04, 0.05, 0.73, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.00],
    [0.02, 0.03, 0.04, 0.05, 0.71, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00],
    [0.01, 0.02, 0.03, 0.04, 0.05, 0.70, 0.05, 0.04, 0.03, 0.02, 0.01],
    [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.71, 0.05, 0.04, 0.03, 0.02],
    [0.00, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.73, 0.05, 0.04, 0.03],
    [0.00, 0.00, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.76, 0.05, 0.04],
    [0.00, 0.00, 0.00, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.80, 0.05],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.85],
])


def softmax(dist):
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output


def log_stable(arr):
    epsilon = 1e-16
    return np.log(arr + epsilon)


# Create transition matrix B, and the matrix A would be explicitly created later
def create_B_matrix():
    B = np.zeros((len(states), len(states), len(actions)))

    for action_id, action_label in enumerate(actions):
        for curr_state, imagstate in enumerate(states):
            x = imagstate

            if action_label == "UP":
                next_x = np.where(x < 8, x + 1, x)
            elif action_label == "DOWN":
                next_x = np.where(x > 0, x - 1, x)
            elif action_label == "STAY":
                next_x = x

            new_state = next_x
            next_state = states.index(next_x)
            B[next_state, curr_state, action_id] = 1.0

    return B


# Get approxiamate posterior
def infer_states(observation_index, A, prior):
    log_likelihood = log_stable(A[observation_index, :])
    log_prior = log_stable(prior)
    qs = softmax(log_likelihood + log_prior)
    return qs


# Calculate $P(s|s,pi)$
def get_expected_states(B, qs_current, action):
    qs_u = B[:, :, action].dot(qs_current)
    return qs_u


# $P(o|qs_u)$
def get_expected_observations(A, qs_u):
    qo_u = A.dot(qs_u)
    return qo_u


def entropy(A):
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A


def kl_divergence(a, b):
    return (log_stable(a) - log_stable(b)).dot(a)


# Calculate expected free energy G
def calculate_G(A, B, C, qs_current, actions):
    G = np.zeros(len(actions))  # vector of expected free energies, one per action

    H_A = entropy(A)  # entropy of the observation model, P(o|s)

    for action_i in range(len(actions)):
        qs_u = get_expected_states(B, qs_current,
                                   action_i)  # expected states, under the action we're currently looping over
        qo_u = get_expected_observations(A,
                                         qs_u)  # expected observations, under the action we're currently looping over

        pred_uncertainty = H_A.dot(qs_u)  # predicted uncertainty, i.e. expected entropy of the A matrix
        pred_div = kl_divergence(qo_u, C)  # predicted divergence

        G[action_i] = pred_uncertainty + pred_div  # sum them together to get expected free energy

    return G


# Calculate alternative policies
def construct_policies(num_states, num_controls=None, policy_len=1, control_fac_idx=None):
    num_factors = len(num_states)
    if control_fac_idx is None:
        if num_controls is not None:
            control_fac_idx = [f for f, n_c in enumerate(num_controls) if n_c > 1]
        else:
            control_fac_idx = list(range(num_factors))

    if num_controls is None:
        num_controls = [num_states[c_idx] if c_idx in control_fac_idx else 1 for c_idx in range(num_factors)]

    x = num_controls * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))
    for pol_i in range(len(policies)):
        policies[pol_i] = np.array(policies[pol_i]).reshape(policy_len, num_factors)

    return policies


# For every alternative policy, calculate its expected free energy
def calculate_G_policies(A, B, C, qs_current, policies):
    G = np.zeros(len(policies))  # initialize the vector of expected free energies, one per policy
    H_A = entropy(A)  # can calculate the entropy of the A matrix beforehand, since it'll be the same for all policies

    for policy_id, policy in enumerate(policies):
        # loop over policies - policy_id will be the linear index of the policy (0, 1, 2, ...) and `policy` will be a
        # column vector where `policy[t,0]` indexes the action entailed by that policy at time `t`

        t_horizon = policy.shape[0]  # temporal depth of the policy

        G_pi = 0.0  # initialize expected free energy for this policy

        for t in range(t_horizon):  # loop over temporal depth of the policy

            action = policy[t, 0]  # action entailed by this particular policy, at time `t`

            # get the past predictive posterior - which is either your current posterior at the current time (not the policy time) or the predictive posterior entailed by this policy, one timstep ago (in policy time)
            if t == 0:
                qs_prev = qs_current
            else:
                qs_prev = qs_pi_t

            # expected states, under the action entailed by the policy at this particular time
            qs_pi_t = get_expected_states(B, qs_prev, action)
            # expected observations, under the action entailed by the policy at this particular time
            qo_pi_t = get_expected_observations(A, qs_pi_t)
            # Kullback-Leibler divergence between expected observations and the prior preferences C
            kld = kl_divergence(qo_pi_t, C)

            G_pi_t = H_A.dot(qs_pi_t) + kld  # predicted uncertainty + predicted divergence, for this policy & timepoint

            G_pi += G_pi_t  # accumulate the expected free energy for each timepoint into the overall EFE for the policy

        G[policy_id] += G_pi

    return G


def compute_prob_actions(actions, policies, Q_pi):
    P_u = np.zeros(len(actions))  # initialize the vector of probabilities of each action

    for policy_id, policy in enumerate(policies):
        P_u[int(policy[0, 0])] += Q_pi[
            policy_id]
        # get the marginal probability for the given action, entailed by this policy at the first timestep

    P_u = utils.norm_dist(P_u)  # normalize the action probabilities

    return P_u


# Simulating the T-steps active inference process with "policy_len"-steps planning
def active_inference_with_planning(A, B, C, D, obs_idx, n_actions, env, policy_len, T):
    """ Initialize prior, first observation, and policies """

    prior = D  # initial prior should be the D vector
    if obs_idx == None:  # if obs is not provided, reset the environment
        obs = env.reset()
    else:
        obs = observation[obs_idx]

    policies = construct_policies([n_states], [n_actions], policy_len=policy_len)

    for t in range(T):
        # convert the observation into the agent's observational state space (in terms of 0 through 8)
        obs_idx = observation.index(obs)

        # perform inference over hidden states
        qs_current = infer_states(obs_idx, A, prior)

        # calculate expected free energy of actions
        G = calculate_G_policies(A, B, C, qs_current, policies)
        # to get action posterior, we marginalize P(u|pi) with the probabilities of each policy Q(pi), given by \sigma(-G)
        Q_pi = softmax(-G)

        # compute the probability of each action
        P_u = compute_prob_actions(actions, policies, Q_pi)

        # sample action from probability distribution over actions
        chosen_action = utils.sample(P_u)

        # compute prior for next timestep of inference
        prior = B[:, :, chosen_action].dot(qs_current)

        # step the generative process and get new observation
        action_label = actions[chosen_action]
        obs = env.step(action_label)

    # get the final observation index
    obs_idx = observation.index(obs)
    # infer the final state
    qs_current = infer_states(obs_idx, A, prior)
    # plot the beliefs of final state

    # calculate actual free energy
    dkl = kl_divergence(qs_current, A[obs_idx, :])
    evidence = log_stable(prior)
    F = dkl - evidence

    return obs_idx, qs_current, F


# After policy taken, the divergence of current states and original preferences is calculated as residual free energy
def residual(C, qs):
    R = kl_divergence(C, qs)
    return R


# updating priors based on residual free energy
def D_update(D, F, R, w):
    if w is None:
        w = 1
    p = np.argmin(F)  # sample the item with minimal actual free energy
    F[p] = F[p] + w * R  # add the residual free energy to this item, forming a new variational free energy
    D = softmax(-F)  # update priors vector over current state according to the new F
    return D


# Change the preference vector C according to observations dynamically
def change_C(obs_idx):
    C = utils.onehot(observation.index(obs_idx), n_observation)
    return C
