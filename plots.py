# import matplotlib.pyplot as plt

# # Data for greedy and epsgreedy strategies
# contexts = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)]
# greedy_avg_prices = [54.22, 51.08, 28.12, 28.11, 53.81, 37.99, 28.09, 28.09]
# epsgreedy_avg_prices = [54.70, 50.84, 28.07, 26.81, 54.21, 39.94, 26.70, 26.70]
# greedy_optimal_prices = [50.14, 35.12, 25.07, 40.59, 45.02, 20.01, 30.08, 55.11]

# # Parameters for plotting
# bar_width = 0.35
# index = range(len(contexts))
# soft_blue = '#a6c1ee'
# soft_green = '#a8dba8'

# plt.figure(figsize=(15,8))

# # Bar chart for average best prices with softer colors
# plt.bar(index, greedy_avg_prices, bar_width, color=soft_blue, label='Greedy Avg Prices')
# plt.bar([i+bar_width for i in index], epsgreedy_avg_prices, bar_width, color=soft_green, label='EpsGreedy Avg Prices')

# # Scatter plot for optimal prices per context at the center of each bar group
# plt.scatter([i+bar_width/2 for i in index], greedy_optimal_prices, color='red', marker='o', label='Optimal Prices', s=100, zorder=3)

# # Formatting
# plt.xlabel('Context')
# plt.ylabel('Prices')
# plt.title('Average Best Prices vs Optimal Prices by Context')
# plt.xticks([i+bar_width/2 for i in index], contexts)
# plt.ylim(0, max(max(greedy_avg_prices), max(epsgreedy_avg_prices)) + 10)
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.grid(axis='both', linestyle='--', linewidth=0.5)

# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression

# # Parameters
# num_records = 10000
# values = [0, 1, 2, 3]
# probabilities = [0.01, 0.08, 0.95, 0.97]
# probabilities = [0.1, 0.8, 0.5, 0.3]

# # Generate dataset
# features = []
# labels = []
# for value, prob in zip(values, probabilities):
#     n_value = num_records // len(values)
#     features.extend([value] * n_value)
#     labels.extend(np.random.choice([0, 1], size=n_value, p=[1-prob, prob]))

# # Create DataFrame
# df = pd.DataFrame({'Feature': features, 'Label': labels})

# # Train logistic regression model
# model = LogisticRegression()
# model.fit(df[['Feature']], df['Label'])

# # Predict probabilities for visualization
# X_test = np.linspace(-1, 4, 300).reshape(-1, 1)
# y_pred = model.predict_proba(X_test)[:, 1]

# # Plot
# plt.figure(figsize=(10, 6))
# plt.scatter(df['Feature'], df['Label'], alpha=0.1, label="Data Points", color='blue')
# plt.plot(X_test, y_pred, color='red', label="Logistic Curve")
# plt.scatter(values, probabilities, color='green', s=100, zorder=5, label="Actual Probabilities")
# plt.title("Logistic Regression Curve with Actual Probabilities")
# plt.xlabel("Feature Value")
# plt.ylabel("Probability")
# plt.legend()
# plt.grid(True)
# plt.show()


import matplotlib.pyplot as plt

# Data for greedy and epsgreedy strategies using dectree model
contexts = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)]
greedy_avg_prices = [42.66, 31.82, 24.27, 36.55, 39.77, 21.43, 27.93, 43.38]
epsgreedy_avg_prices = [48.49, 36.30, 25.52, 41.16, 44.91, 20.45, 31.23, 50.80]
greedy_optimal_prices = [50.14, 35.12, 25.07, 40.59, 45.02, 20.01, 30.08, 55.11]

# Parameters for plotting
bar_width = 0.35
index = range(len(contexts))
soft_blue = '#a6c1ee'
soft_green = '#a8dba8'

plt.figure(figsize=(15,8))

# Bar chart for average best prices with softer colors
plt.bar(index, greedy_avg_prices, bar_width, color=soft_blue, label='Greedy (dectree) Avg Prices')
plt.bar([i+bar_width for i in index], epsgreedy_avg_prices, bar_width, color=soft_green, label='EpsGreedy (dectree) Avg Prices')

# Scatter plot for optimal prices per context at the center of each bar group
plt.scatter([i+bar_width/2 for i in index], greedy_optimal_prices, color='red', marker='o', label='Optimal Prices', s=100, zorder=3)

# Formatting
plt.xlabel('Context')
plt.ylabel('Prices')
plt.title('Average Best Prices vs Optimal Prices by Context for dectree Model')
plt.xticks([i+bar_width/2 for i in index], contexts)
plt.ylim(0, max(max(greedy_avg_prices), max(epsgreedy_avg_prices)) + 10)
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid(axis='both', linestyle='--', linewidth=0.5)

plt.show()
