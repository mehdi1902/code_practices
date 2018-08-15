import gym
import numpy as np
from gym import wrappers

def get_action(state, w):
    out = 1 if state.dot(w)>0 else 0
    # print out
    return out

def play_one_episode(env, params, max_iter=10000):
    observation = env.reset()
    done = False
    i = 0
    while done == False and i < max_iter:
        i += 1
        observation, r, done, _ = env.step(get_action(observation, params))
        if done:
            break
    return i

def play_multiple_episodes(env, N, params, max_iter=10000):
    sum_value = 0
    for i in range(N):
        sum_value += play_one_episode(env, params, max_iter)
    return sum_value / float(N)

def random_search(env, n_episodes, n_iters, max_iter=10000):
    params = np.random.random(4)*2 - 1
    best = 0
    for i in range(n_episodes):
        avg_value = play_multiple_episodes(env, n_iters, params, max_iter)
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
    