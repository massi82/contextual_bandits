import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from demand import generate_context, get_reward, prices, optimal_prices, get_price_probability
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import defaultdict

class Model():
    def __init__(self, type="dectree"):
        self.type = type        
        if type == "dectree":
            self.model = DecisionTreeClassifier()
        elif type == "logistic":
            learning_rate = "adaptive"
            penalty = "l2"
            alpha = 0.01
            shuffle = True
            eta0 = 0.01
            self.model = SGDClassifier(loss="log_loss", learning_rate=learning_rate, penalty=penalty, 
                                       alpha=alpha, shuffle=shuffle, eta0=eta0, random_state=42)
        elif type == "randomforest":
            self.model = RandomForestClassifier(
                n_estimators=10,  # Reduced number of trees
                max_depth=5,      # Reduced depth of each tree
                max_samples=0.5,  # Using 50% of the dataset for bootstrapping
                n_jobs=1,        # Use all available cores for parallelization
                max_features=None,
                random_state=42   # Ensuring reproducibility    
            )

        self.X = np.empty((0, 2))
        self.y = np.empty((0,))

    def add_sample(self, X, y):
        self.X = np.vstack((self.X, X))
        self.y = np.hstack((self.y, y))

    def can_predict(self):
        return np.any(self.y) and not np.all(self.y)

    def get_prob(self, X):
        return self.model.predict_proba([X])[0][1]

    def fit(self, classes=[0, 1]):
        if self.type in ["dectree", "randomforest"]:
            self.model.fit(self.X, self.y)
        elif self.type == "logistic":
            self.model.partial_fit([self.X[-1]], [self.y[-1]], classes=classes)

    def get_averages(self):
        df = pd.DataFrame({
            'X': [tuple(x) for x in self.X],
            'y': self.y
        })

        # Group by 'X' and calculate mean of 'y'        
        result = df.groupby('X').y.agg(['mean', 'count']).reset_index()
        print(result)
        return result
    
    def get_context_mean(self, X):
        i = 0
        tot = 0
        for x, y in zip(self.X, self.y):
            if tuple(x) == X:
                tot += y
                i += 1

        return tot/i
    
    def get_variance(self, X):
        if self.type != "randomforest":
            raise ValueError("Variance retrieval is implemented only for Random Forest")
        # Get predictions from all trees in the forest for X
        predictions = np.array([tree.predict([X]) for tree in self.model.estimators_])
        # Return the variance of predictions
        return np.var(predictions)
    
def greedy(models, X, prices):         
    probabilities = [model.get_prob(X)*prices[i] for i, model in enumerate(models)]
    return np.argmax(probabilities)

def epsilon_greedy(models, X, prices, epsilon=0.1):            
    probabilities = [model.get_prob(X)*prices[i] for i, model in enumerate(models)]
    best = np.argmax(probabilities) 
    
    if np.random.rand() < epsilon:
        other_choices = np.delete(np.arange(len(prices)), best)
        return np.random.choice(other_choices)
    else:
        return best

def UCB1(models, X, prices, iteration, arm_counter, C=0.7): 
    # Total number of times any arm has been played
    total_plays = iteration + 1 

    probs = np.array([model.get_prob(X) for model in models])
    max_prob = probs.max()
    if max_prob == 0:
        return np.random.choice(len(prices))
    norm_probs = probs/max_prob
    # Calculate upper bounds for all arms
    ucb_values = norm_probs + C * np.sqrt(2 * np.log(total_plays) / arm_counter)
    ucb_values *= max_prob  

    ucb_values = [ucb_values[i]*prices[i] for i in range(len(prices))]
      
    return np.argmax(ucb_values)

def run_simulation(prices, nstep, strategy="epsgreedy", model_type="logistic"):
    narm = len(prices)
    regret = 0
    cum_regret = np.zeros((nstep,))    
    # Create a model for each arm (price)
    models = [Model(model_type) for i in range(narm)]
    # Keep track of how many times we hit the optimal price
    optimal_choice = 0

    # Simulation loop
    for iteration in range(nstep):
        # Generate a random context for the customer
        X, a, b = generate_context()

        if strategy == "random":
            arm = np.random.choice(len(prices))
        elif not all([model.can_predict() for model in models]):
            # manage cold start
            for i, model in enumerate(models):
                if not model.can_predict():
                    arm = i
                    break
        elif strategy == "greedy":
            arm = greedy(models, X, prices)
        elif strategy == "epsgreedy":
            arm = epsilon_greedy(models, X, prices, epsilon=0.1)
        
        # Simulate the customer's response
        reward = get_reward(prices[arm], a, b)
        # Compute regret using the known optimal_price
        optimal_reward = get_reward(optimal_prices[X], a, b)
        regret = optimal_prices[X]*optimal_reward - prices[arm]*reward
        cum_regret[iteration] = cum_regret[iteration-1]+regret
        if prices[arm] == int(optimal_prices[X]):
            optimal_choice += 1

        if strategy == "random":
            continue
        # Update the model
        models[arm].add_sample(X, reward)
        models[arm].fit()
                 
        # if all([model.can_predict() for model in models]) and X == (0, 0):
        #     probabilities = np.array([model.get_prob(X) for i, model in enumerate(models)])
        #     real_price_probs = np.array([get_price_probability((0, 0), p) for p in prices])
        #     print(real_price_probs-probabilities)

    return models, cum_regret, optimal_choice/nstep

# Simulation parameters
nepoch = 1000
nstep = 10000
regret_curves = {}
random_done = False
for model_type in ["logistic"]:#, "dectree"
    for strategy in ["random", "greedy", "epsgreedy"]:
        if strategy == "random" and random_done:
            continue
        elif strategy == "random":
            random_done = True 
        curve = model_type+"-"+strategy if strategy != "random" else "random"
        regret_curves[curve] = np.zeros((nstep,)) 
        regrets = []
        optimal_chioces = []
        best_prices = defaultdict(list)
        for ep in range(nepoch):
            if ep%100 == 0:
                print(ep)
            models, regret, optimal_chioce = run_simulation(prices, nstep, strategy=strategy, model_type=model_type)
            regret_curves[curve] += regret
            regrets.append(regret[-1]) 
            optimal_chioces.append(100*optimal_chioce)  
            if strategy != "random":    
                for geo in (0, 1):
                    for age in (0, 1, 2, 3):
                        probs = [model.get_prob((geo, age))*prices[i] for i, model in enumerate(models)]
                        best_price = prices[np.argmax(probs)]        
                        best_prices[(geo, age)].append(best_price)
        regret_curves[curve] /= nepoch

        if strategy != "random":    
            print("-------------\nStrategy: %s, Model: %s" %(strategy, model_type)) 
            for k in best_prices:
                best_prices[k] = np.array(best_prices[k])
                print("Average best price for context %s is %.2f (std %.2f, optimal %.2f)" %(k, np.mean(best_prices[k]), 
                                                                                             np.std(best_prices[k]), 
                                                                                             optimal_prices[k]))
        else:   
            print("-------------\nStrategy: %s" %(strategy)) 
        print("Regret -> mean: %.2f, median: %.2f, std: %.2f" %(np.mean(regrets), np.median(regrets), np.std(regrets)))
        print("Optimal choice -> mean: %.2f%%, median: %.2f%%, std: %.2f%%" %(np.mean(optimal_chioces), np.median(optimal_chioces), np.std(optimal_chioces)))

plt.figure(figsize=(12, 6))
for label in regret_curves:
    plt.plot(regret_curves[label], label=label)
plt.xlabel("Time Step")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret Comparison")
plt.legend()
plt.grid(True)
plt.show()