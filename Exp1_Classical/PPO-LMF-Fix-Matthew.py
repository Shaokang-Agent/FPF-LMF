import numpy as np
from keras.utils import to_categorical
import copy
from common.utils import eligibility_traces, default_config, make_env, discount_rewards
from common.df_ppo import PPOPolicyNetwork, ValueNetwork
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
render = False
normalize_inputs = True

config = default_config()
LAMBDA = float(config['agent']['lambda'])
lr_actor = float(config['agent']['lr_actor'])
twophase_proportion = float(config['agent']['twophase_proportion'])


env = make_env(config, normalize_inputs)
env.toggle_compute_neighbors()

n_agent = env.n_agent
T = env.T
GAMMA = env.GAMMA
if config['agent']['env_name'] == "matthew":
   n_episode = 40000
   state_input_size = 130
elif config['agent']['env_name'] == "plant":
    n_episode = 20000
    if config['agent']['plant_ohe']:
        state_input_size = 5*5+3*12*12
    else:
        state_input_size = 5*5+12*12
else:
    n_episode = 10000
    state_input_size = 74
max_steps = env.max_steps
n_actions = env.n_actions

alpha = 1e-3
beta = 1.5
if config['agent']['env_name'] == "job":
    fed_epi = 5
else:
    fed_epi = 10
round = 5
reward_step = np.zeros([round, n_episode, n_agent])
if not os.path.exists("./" + config['agent']['env_name']):
    os.makedirs("./" + config['agent']['env_name'])

for round_num in range(round):
    writter = SummaryWriter("./" + config['agent']['env_name'] + "/runs/PPO-LMF-Fix-Matthew" + "-fed_epi" + str(fed_epi) + "/" +str(round_num))
    DPi = []
    DV = []

    for i in range(n_agent):
        DPi.append(PPOPolicyNetwork(name="Ego", num_features=env.input_size, num_actions=n_actions, layer_size=256, epsilon=0.1,
                                   learning_rate=lr_actor, alpha=alpha, beta=beta, lamdaw=0, lamdae=0))
        DV.append(ValueNetwork(num_features=env.input_size, hidden_size=256, learning_rate=0.001))

    memory_ep_rewards = [deque() for _ in range(n_agent)]
    average_jpi = np.zeros(n_agent)

    for i_episode in tqdm(range(n_episode)):
        memory_ep_rewards = [deque() for _ in range(n_agent)]
        average_jpi = np.zeros(n_agent)

        avg = [0.] * n_agent

        ep_actions = [[] for _ in range(n_agent)]
        ep_rewards = [[] for _ in range(n_agent)]
        ep_obss = [[] for _ in range(n_agent)]
        ep_states = []
        ep_states_h = []

        score = 0
        steps = 0
        su = [0.] * n_agent
        su = np.array(su)

        env_state, obs = env.reset()

        done = False
        while steps < max_steps and not done:
            steps += 1
            action = []
            ep_states.append(env_state)
            state_h = copy.deepcopy(env_state)
            more_return = average_jpi
            more_return = (more_return - np.mean(more_return)) / (np.std(more_return) + 0.0000000001)
            state_h.extend(more_return)
            for i in range(n_agent):
                # add more information
                more_action = DPi[i].get_dist(np.array([obs[i]]))[0]
                state_h.extend(more_action)
                action.append(np.random.choice(range(n_actions), p=more_action))
                ep_actions[i].append(to_categorical(action[i], n_actions))
                ep_obss[i].append(obs[i])

            ep_states_h.append(state_h)
            env_state, obs, rewards, done = env.step(action)

            su += np.array(rewards)
            score += sum(rewards)

            for i in range(n_agent):
                ep_rewards[i].append(rewards[i])
                memory_ep_rewards[i].append(rewards[i])
                average_jpi[i] += rewards[i]
                if len(memory_ep_rewards[i]) > max_steps * 5:
                    average_jpi[i] -= memory_ep_rewards[i].popleft()

            if steps % T == 0:
                D_all_ep_advantages = []
                Futi_all_ep_advantages = []
                Fega_all_ep_advantages = []

                ep_actions = np.array(ep_actions)
                ep_rewards = np.array(ep_rewards, dtype=np.float_)
                ep_states = np.array(ep_states)
                ep_states_h = np.array(ep_states_h)
                ep_rewards_mean = ep_rewards.mean(axis=0)
                ep_returns_std = np.zeros(T)
                for t_index in range(T):
                    ep_returns_std[t_index] = -np.std(ep_rewards[:,:t_index+1].sum(axis=1))
                nstate = copy.deepcopy(env_state)
                more_return = average_jpi
                more_return = (more_return - np.mean(more_return)) / (np.std(more_return) + 0.0000000001)
                nstate.extend(more_return)
                for i in range(n_agent):
                    more_action = DPi[i].get_dist(np.array([obs[i]]))[0]
                    nstate.extend(more_action)
                for i in range(n_agent):
                    if LAMBDA < -0.1:
                        # Decentralized
                        D_targets = discount_rewards(ep_rewards[i], GAMMA)
                        DV[i].update(ep_obss[i], D_targets)
                        D_vs = DV[i].get(ep_obss[i])
                    else:
                        # Decentralized
                        D_vs = DV[i].get(ep_obss[i])
                        D_targets = eligibility_traces(ep_rewards[i], D_vs, DV[i].get([obs[i]]), GAMMA, LAMBDA)
                        DV[i].update(ep_obss[i], D_targets)
                    # Decentralized
                    D_ep_advantages = D_targets - D_vs
                    D_ep_advantages = (D_ep_advantages - np.mean(D_ep_advantages)) / (np.std(D_ep_advantages) + 0.0000000001)
                    D_all_ep_advantages.append(D_ep_advantages)

                # Decentralized
                D_all_ep_advantages = np.array(D_all_ep_advantages)

                for i in range(n_agent):
                    DPi[i].update(ep_obss[i], ep_actions[i], D_all_ep_advantages[i])

                if (i_episode + 1) % fed_epi == 0 and steps == T:
                    DPi_gradients = []
                    for i in range(n_agent):
                        DPi_gradients.append(DPi[i].broad_gradients())
                    for i in range(n_agent):
                        if i < n_agent // 2:
                            DPi[i].update_consensus_gradients(DPi_gradients[:n_agent // 2])
                        else:
                            DPi[i].update_consensus_gradients(DPi_gradients[n_agent // 2:])

                ep_actions = [[] for _ in range(n_agent)]
                ep_rewards = [[] for _ in range(n_agent)]
                ep_obss = [[] for _ in range(n_agent)]
                ep_states = []
                ep_states_h = []

            if render:
                env.render()

        print(config['agent']['env_name'], "PPO-LMF-Fix-Matthew" + "-fed_epi" + str(fed_epi), round_num, i_episode)
        print(su, su.sum())
        writter.add_scalar("Total Rewards", su.sum(), i_episode)
        reward_step[round_num, i_episode] = su
        np.save("./" + config['agent']['env_name'] + "/PPO-LMF-Fix-Matthew" + "-fed_epi" + str(fed_epi) + ".npy", reward_step)




