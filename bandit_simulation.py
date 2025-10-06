# bandit_simulation.py
import numpy as np
import matplotlib.pyplot as plt

print("--- Multi-Armed Bandit Simulation using Thompson Sampling ---")

# --- 1. Define the Simulation Environment ---

# Let's say we're choosing between three of our models
# Each is an "arm" of the bandit
arms = ['SVD', 'Sequential', 'Autoencoder']
n_arms = len(arms)

# Define the TRUE click-through rate (CTR) for each model.
# The algorithm does NOT know this. This is our ground truth.
# In this scenario, the Sequential model is the best.
true_ctrs = {
    'SVD': 0.10,         # 10% chance of a click
    'Sequential': 0.15,  # 15% chance of a click (the best arm)
    'Autoencoder': 0.12  # 12% chance of a click
}

# Number of users to simulate
n_trials = 10000

# --- 2. Initialize Thompson Sampling Parameters ---

# We use the Beta distribution, which is defined by two parameters: alpha and beta.
# alpha can be thought of as "successes + 1"
# beta can be thought of as "failures + 1"
# We start with 1 for both, representing no prior knowledge (a uniform distribution).
beta_params = {arm: {'alpha': 1, 'beta': 1} for arm in arms}

# --- 3. Run the Simulation ---

# Lists to store the history of our simulation
arms_chosen = []
rewards_received = []

for i in range(n_trials):
    # --- a. Sampling Step ---
    # Draw a sample from each arm's current Beta distribution
    sampled_values = {
        arm: np.random.beta(params['alpha'], params['beta'])
        for arm, params in beta_params.items()
    }
    
    # --- b. Selection Step (Exploration/Exploitation) ---
    # Choose the arm with the highest sampled value
    chosen_arm = max(sampled_values, key=sampled_values.get)
    arms_chosen.append(chosen_arm)
    
    # --- c. Reward Step (Simulating Reality) ---
    # Simulate a user click based on the chosen arm's TRUE CTR
    if np.random.rand() < true_ctrs[chosen_arm]:
        reward = 1
    else:
        reward = 0
    rewards_received.append(reward)
    
    # --- d. Update Step (Learning) ---
    # Update the Beta distribution parameters for the chosen arm
    if reward == 1:
        beta_params[chosen_arm]['alpha'] += 1
    else:
        beta_params[chosen_arm]['beta'] += 1


# --- 4. Analyze Results ---
total_reward = sum(rewards_received)
overall_ctr = total_reward / n_trials

print(f"\nTotal Reward Earned: {total_reward}")
print(f"Overall Click-Through Rate: {overall_ctr:.2%}")

# Calculate how many times each arm was chosen
from collections import Counter
choice_counts = Counter(arms_chosen)

print("\nArm Selection Frequencies:")
for arm in arms:
    pulls = choice_counts.get(arm, 0)
    print(f"- {arm}: Chosen {pulls} times ({pulls/n_trials:.2%})")

# --- 5. Visualize the Learning Process ---

# Calculate the percentage of times each arm was chosen over windows of the simulation
window_size = 500
selection_history = {arm: [] for arm in arms}
for i in range(0, n_trials, window_size):
    window = arms_chosen[i : i + window_size]
    for arm in arms:
        selection_history[arm].append(window.count(arm) / window_size * 100)

plt.figure(figsize=(12, 7))
for arm in arms:
    plt.plot(selection_history[arm], label=f'Model: {arm} (True CTR: {true_ctrs[arm]:.0%})')

plt.title('Thompson Sampling: Percentage of Times Each Model was Chosen', fontsize=16)
plt.xlabel(f'Simulation Window (x{window_size} users)')
plt.ylabel('Percentage Chosen (%)')
plt.legend()
plt.grid(True)
plt.savefig('bandit_simulation_results.png')
print("\nGenerated plot: 'bandit_simulation_results.png'")
plt.show()


