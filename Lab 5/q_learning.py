import gym
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

def evaluate_policy(env, q, eval_runs=50):
    total = 0
    for _ in range(eval_runs):
        s, _ = env.reset()
        done = False
        while not done:
            a = np.argmax(q[s, :])
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
    return total / eval_runs


def e_greedy(s, q, eps, nr_A):
    prob = np.random.rand()
    if prob < eps:
        return np.random.randint(0, nr_A)
    else:
        return np.argmax(q[s,:])


def q_learning(env, epoch, gamma=0.5, eps=0.1, alpha=0.1, eval_interval=1000):

    nr_S = env.observation_space.n
    nr_A = env.action_space.n

    policy = np.zeros(nr_S, dtype=np.int64)

    q = np.zeros((nr_S, nr_A))

    train_rewards = []
    eval_rewards = []
    
    for e in range(epoch):
        s, _ = env.reset()
        final = False
        total_reward = 0

        while not final:
            a = e_greedy(s, q, eps, nr_A)

            s_prime, reward, terminated, truncated, _ = env.step(a)
            final = terminated or truncated

            q[s, a] += alpha*(reward + gamma * np.max(q[s_prime, :]) - q[s,a])
        
            total_reward += reward
            s = s_prime

        train_rewards.append(total_reward)

        if e % eval_interval == 0:
            avg_eval_reward = evaluate_policy(env, q)
            eval_rewards.append((e, avg_eval_reward))

    print(q)
    for s in range(nr_S):
        policy[s] = np.argmax(q[s, :])

    return policy, train_rewards, eval_rewards
    
def sarsa(env, epoch, gamma=0.5, eps=0.1, alpha=0.1, eval_interval=1000):
    nr_S = env.observation_space.n
    nr_A = env.action_space.n

    policy = np.zeros(nr_S, dtype=np.int64)

    q = np.zeros((nr_S, nr_A))

    train_rewards = []
    eval_rewards = []


    for e in range(epoch):
        s, _ = env.reset()
        final = False
        total_reward = 0

        a = e_greedy(s, q, eps, nr_A)
        while not final:
            s_prime, reward, terminated, truncated, _ = env.step(a)
            final = terminated or truncated

            a_prime = e_greedy(s_prime, q, eps, nr_A)

            q[s, a] += alpha*(reward + gamma * q[s_prime, a_prime] - q[s,a])

            s = s_prime
            a = a_prime

            total_reward += reward

        train_rewards.append(total_reward)

        if e % eval_interval == 0:
            avg_eval_reward = evaluate_policy(env, q)
            eval_rewards.append((e, avg_eval_reward))

    for s in range(nr_S):
        policy[s] = np.argmax(q[s, :])

    return policy, train_rewards, eval_rewards

def plot_results(q_train_rewards, sarsa_train_rewards, q_eval_rewards, sarsa_eval_rewards):
    plt.figure(figsize=(12, 6))
    window = 100
    q_smooth = np.convolve(q_train_rewards, np.ones(window)/window, mode='valid')
    s_smooth = np.convolve(sarsa_train_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(q_smooth, label='Q-Learning (train)', alpha=0.7)
    plt.plot(s_smooth, label='SARSA (train)', alpha=0.7)

    q_epochs, q_evals = zip(*q_eval_rewards)
    s_epochs, s_evals = zip(*sarsa_eval_rewards)
    plt.scatter(q_epochs, q_evals, label='Q-Learning (eval)', marker='x')
    plt.scatter(s_epochs, s_evals, label='SARSA (eval)', marker='o')
    
    plt.title("Comparison of Q-Learning and SARSA")
    plt.xlabel("Training Epochs")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()




#env = gym.make("Taxi-v3")
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)


epochs = 10000
gamma = 0.99
eps = 0.3       #increase eps : increase exploration
alpha = 0.8           # Faster learning (sparse rewards need aggressive updates)

policy_q, q_train_rewards, q_eval_rewards = q_learning(env, epoch=epochs, gamma=gamma, alpha=alpha, eps=eps)
policy_sarsa, sarsa_train_rewards, sarsa_eval_rewards = sarsa(env, epoch=epochs, gamma=gamma, alpha=alpha, eps=eps)

print("DONE")

plot_results(q_train_rewards, sarsa_train_rewards, q_eval_rewards, sarsa_eval_rewards)

#env = gym.make("Taxi-v3", render_mode="human")
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode='human')


print(policy_sarsa)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = policy_sarsa[observation]

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()


