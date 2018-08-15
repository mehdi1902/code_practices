import gym
import numpy as np
from gym import wrappers


def quantize(observation):
    return int(observation[0])

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
        return np.argmax(Q[s_bin])


def play_one_episode(env, Q, epsilon, max_iter=10000):
    observation = env.reset()
    done = False
    i = 0
    while done == False and i < max_iter:
        i += 1
        action = get_action(observation, [0, 1], Q, epsilon)
        observation, r, done, _ = env.step(action)
        if done:
            break
    return i

def play_multiple_episodes(env, N, params, Q, epsilon, max_iter=10000):
    sum_value = 0
    for i in range(N):
        sum_value += play_one_episode(env, params, max_iter)
    return sum_value / float(N)


def q_learning(env, n_episodes, n_iters, max_iter=1000, epsilon=.1):
    Q = {i:[0, 0] for i in range(-5, 6, 1)}
    for i in range(n_episodes):
        avg_value = play_multiple_episodes(env, n_iters, params, Q, epsilon, max_iter)
#         print avg_value
        if avg_value > best:
            best = avg_value
            print 'best is %f (iteration %i)' % (best, i)
            best_params = params
    return best_params, best


def random_search(env, n_episodes, n_iters, max_iter=10000):
    params = np.random.random(4)*2 - 1
    best = 0
    for i in range(n_episodes):
        avg_value = play_multiple_episodes(env, n_episodes, n_iters, params, max_iter)
#         print avg_value
        if avg_value > best:
            best = avg_value
            print 'best is %f (iteration %i)' % (best, i)
            best_params = params
    return best_params, best


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    best_params, best_value = random_search(env, 1000, 100)
    print "best value was %f" % best_value

    env = wrappers.Monitor(env, 'videos', force=True)
    play_one_episode(env, best_params)
    