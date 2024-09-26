import numpy as np
import matplotlib.pyplot as plt

# Define the reaction functions
def reaction_function_1(q2, alpha, beta, c):
    return (alpha - beta * q2 - c) / (2 * beta)

def reaction_function_2(q1, alpha, beta, c):
    return (alpha - beta * q1 - c) / (2 * beta)

# Generate values for q1 and q2
q_values = np.linspace(0, 10, 100)  # Adjust the range based on your problem

# Plot the reaction functions for various parameter values
alpha_values = [8, 10, 12]  # You can add more values if needed
beta_values = [1, 1.5, 2]   # You can add more values if needed
c_values = [1, 2, 3]        # You can add more values if needed

plt.figure(figsize=(8, 6))

for alpha in alpha_values:
    for beta in beta_values:
        for c in c_values:
            q1_values = reaction_function_1(q_values, alpha, beta, c)
            q2_values = reaction_function_2(q_values, alpha, beta, c)
            plt.plot(q_values, q1_values, label=f'Company 1: alpha={alpha}, beta={beta}, c={c}')
            plt.plot(q_values, q2_values, label=f'Company 2: alpha={alpha}, beta={beta}, c={c}')

# Labels and legend
plt.xlabel('Quantity')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.title('Cournot Duopoly: Reaction Functions')
plt.show()
