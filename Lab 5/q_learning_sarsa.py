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


def run_env_graphic(env, policy):
    observation, _ = env.reset(seed=42)
    for _ in range(1000):
        action = policy[observation]

        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset()
            break

    #env.close()



def run_experiment(env, epochs, fixed_gamma, fixed_eps, alphas, algorithm="Q", n_trials=1):
    results = {}
    
    for alpha in alphas:
        results[alpha] = {
            'train_rewards': [],
            'eval_rewards': []
        }
        
        for _ in range(n_trials):
            if algorithm=="Q":
                policy, train_rewards, eval_rewards = q_learning(
                  env, epoch=epochs, gamma=fixed_gamma, eps=fixed_eps, alpha=alpha
                )
            elif algorithm=="SARSA":
                policy, train_rewards, eval_rewards = sarsa(
                    env, epoch=epochs, gamma=fixed_gamma, eps=fixed_eps, alpha=alpha
                )
            results[alpha]['train_rewards'].append(train_rewards)
            results[alpha]['eval_rewards'].append(eval_rewards)
            
    return results

def plot_improved_comparison(results, fixed_eps, fixed_gamma, window=100):
    """Plot improved comparison of different alpha values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Full training history with smoothing
    for alpha, data in results.items():
        # Average over trials
        avg_rewards = np.mean(data['train_rewards'], axis=0)
        
        # Apply smoothing
        smooth_rewards = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')
        # smooth_rewards = data['train_rewards']

        # Plot
        ax1.plot(smooth_rewards, label=f'α={alpha}')
    
    ax1.set_title(f"Full Training History (ε={fixed_eps}, γ={fixed_gamma})")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Zoomed in on early learning
    zoom_range = 5000  # First 5000 epochs
    
    for alpha, data in results.items():
        # Average over trials
        avg_rewards = np.mean(data['train_rewards'], axis=0)
        
        # Apply smoothing
        smooth_rewards = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')
        # smooth_rewards = data['train_rewards']

        # Plot zoomed portion
        x_values = range(len(smooth_rewards))
        zoomed_x = [x for x in x_values if x < zoom_range]
        zoomed_y = smooth_rewards[:min(zoom_range, len(smooth_rewards))]
        
        ax2.plot(zoomed_x, zoomed_y, label=f'α={alpha}')
        
        # Add evaluation points if available
        eval_x = []
        eval_y = []
        for trial in data['eval_rewards']:
            for epoch, reward in trial:
                if epoch < zoom_range:
                    eval_x.append(epoch)
                    eval_y.append(reward)
        
        if eval_x:
            ax2.scatter(eval_x, eval_y, marker='o', alpha=0.3)
    
    ax2.set_title(f"Early Training (ε={fixed_eps}, γ={fixed_gamma})")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Reward")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_varying_alpha(env):
    parameters = {
        'gamma': [0.5, 0.9],
        'epsilon': [0.1, 0.5, 0.8],
        'alpha': [0.1, 0.5, 0.9]
    }
    
    epochs = 20000
    
    for fixed_eps in parameters['epsilon']:
        for fixed_gamma in parameters['gamma']:
            alphas = parameters['alpha']
            
            # Run experiments
            results = run_experiment(
                env, epochs, fixed_gamma, fixed_eps, alphas, algorithm="SARSA"
            )
            
            # Plot improved comparison
            plot_improved_comparison(results, fixed_eps, fixed_gamma)



if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    # env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    
    # run_varying_alpha(env)


    # Used for FrozenLake
    epochs = 30000
    gamma = 0.9
    eps = 0.8 
    alpha = 0.5         

    #Used for taxi
    epochs = 28000
    gamma = 0.8
    eps = 0.3 
    alpha = 0.2     

    policy_q, q_train_rewards, q_eval_rewards = q_learning(env, epoch=epochs, gamma=gamma, alpha=alpha, eps=eps)
    policy_sarsa, sarsa_train_rewards, sarsa_eval_rewards = sarsa(env, epoch=epochs, gamma=gamma, alpha=alpha, eps=eps)

    plot_results(q_train_rewards, sarsa_train_rewards, q_eval_rewards, sarsa_eval_rewards)

    env = gym.make("Taxi-v3", render_mode="human")
    # env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode='human')

    print("Running Q-Learning Policy...")
    run_env_graphic(env, policy_q)
    print("Running SARSA Policy...")
    run_env_graphic(env, policy_sarsa)

