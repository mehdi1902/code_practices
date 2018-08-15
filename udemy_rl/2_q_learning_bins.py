import gym
import numpy as np
from gym import wrappers


def quantize(observation):
    f = []
    f.append(np.linspace(-3.4, 3.4, 11))
    f.append(np.linspace(-50, 50, 11))
    f.append(np.linspace(-0.75, 0.75, 11))
    f.append(np.linspace(-50, 50, 11))
    return [np.where(observation[i]<f[i])[0][0]-1 for i in range(len(observation))]

def get_q(Q, quantized_state):
    i1, i2, i3, i4 = quantized_state
    return Q[i1][i2][i3][i4]

def random_action(action_space, mode='uniform'):
    if not mode == 'uniform':
        print "Other random selections are not implemented yet! So uniform for now"
    return action_space[int(np.random.random() * len(action_space))]

def get_action(observation, action_space, Q, epsilon, random_mode='uniform'):
    e = np.random.random()
    if e < epsilon:
        # Epsilon-greedy
        return random_action(action_space, random_mode)
    else:
        # Choose from the Q
        s_bin = quantize(observation)
        return np.argmax(get_q(Q, s_bin))

def update_q(Q, s, a, s_prime, r, gamma):
    # print r + gamma * (np.max(Q[quantize(s_prime)]))
    s = quantize(s)
    s_prime = quantize(s_prime)
    # print Q[s].shape
    get_q(Q, s)[a] = .9 * get_q(Q, s)[a] + .1 * (r + gamma * (np.max(get_q(Q, s_prime))))
    # print Q[s][a], s, s_prime
    return Q

def play_one_episode(env, Q, epsilon, gamma, max_iter=10000, test=False):
    observation = env.reset()
    done = False
    i = 0
    cum_reward = 0
    while done == False and i < max_iter:
        i += 1
        action = get_action(observation, [0, 1], Q, epsilon)
        prev_state = observation
        observation, r, done, _ = env.step(action)
        
        # print r
        if done and i < max_iter:
            r = -300
        cum_reward += r
        Q = update_q(Q, prev_state, action, observation, cum_reward, gamma)
        # if done and not test:
        #     break
    return i


def q_learning(env, n_episodes, gamma, epsilon, max_iter=1000):
    Q = np.zeros((10, 10, 10, 10, 2))#{i:[0, 0] for i in np.arange(-500, 600, 1)/100.}
    best = 0
    for i in range(n_episodes):
        value = play_one_episode(env, Q, epsilon, gamma, max_iter)
#         print avg_value
        best = max(value, best)
        if not i % 100:
            print 'best is %f (iteration %i)' % (best, i)
    return Q


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    # Q = {i:[0, 0] for i in np.arange(-50, 60, 1)/10.}
    # print Q
    # i, q = play_one_episode(env, Q, .1, .9)
    # print q
    # print i
    Q = q_learning(env, 500, .9, .1)
    # print Q
    # best_params, best_value = random_search(env, 1000, 100)
    # print "best value was %f" % best_value

    env = wrappers.Monitor(env, 'videos', force=True)
    play_one_episode(env, Q, .1, .9, test=True)
    # print Q
    # 