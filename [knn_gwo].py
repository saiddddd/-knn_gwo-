#by : Said Al Afghani Edsa
#date : 28/05/2023

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

np.random.seed(42)

class GreyWolfOptimizer:
    def __init__(self):
        self.search_space = None
        self.population = None
        self.fitness = None
        self.leader_position = None
        self.leader_fitness = None
        self.best_solution = None

    def initialize_search_space(self, lower_bound, upper_bound):
        self.search_space = np.vstack((lower_bound, upper_bound))

    def initialize_population(self, population_size=10):
        num_features = self.search_space.shape[1]
        self.population = np.random.uniform(low=self.search_space[0], high=self.search_space[1], size=(population_size, num_features))
        self.fitness = np.zeros(population_size)

    def calculate_fitness(self, X_train, y_train):
        for i, wolf in enumerate(self.population):
            k = int(round(wolf[0]))
            k = max(1, k)

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            self.fitness[i] = knn.score(X_train, y_train)

    def update_leader_positions(self):
        leader_index = np.argmax(self.fitness)
        self.leader_position = self.population[leader_index]
        self.leader_fitness = self.fitness[leader_index]
        self.best_solution = self.leader_position.copy()

    def update_positions(self, iteration, max_iterations):
        a = 2 - iteration * ((2) / max_iterations)

        num_wolves = self.population.shape[0]
        for i in range(num_wolves):
            A1 = 2 * np.random.rand() - 1
            C1 = 2 * np.random.rand()
            D_alpha = np.abs(C1 * self.leader_position - self.population[i])
            X1 = self.leader_position - A1 * D_alpha

            r1 = np.random.rand()
            A2 = 2 * r1 - 1
            C2 = 2 * np.random.rand()
            D_beta = np.abs(C2 * self.leader_position - self.population[i])
            X2 = self.leader_position - A2 * D_beta

            r2 = np.random.rand()
            A3 = 2 * r2 - 1
            C3 = 2 * np.random.rand()
            D_delta = np.abs(C3 * self.population[i] - self.population[i])
            X3 = self.population[i] - A3 * D_delta

            updated_wolf = (X1 + X2 + X3) / 3
            updated_wolf = np.clip(updated_wolf, self.search_space[0], self.search_space[1])
            self.population[i] = updated_wolf

            # Increment k value
            self.population[i][0] = self.population[i][0] + a

    def get_best_solution(self):
        return self.best_solution


def knn_gwo(X_train, y_train, X_test, y_test, gwo_iterations=10):
    lower_bound = [1, 0.01]
    upper_bound = [10, 0.99]

    gwo = GreyWolfOptimizer()
    gwo.initialize_search_space(lower_bound, upper_bound)
    gwo.initialize_population()

    best_accuracy = 0.0
    best_k = 1

    for iteration in range(gwo_iterations):
        gwo.calculate_fitness(X_train, y_train)
        gwo.update_leader_positions()
        gwo.update_positions(iteration, gwo_iterations)

        best_solution = gwo.get_best_solution()
        k = int(round(best_solution[0]))
        k = max(1, k)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_k = k

        print("Iterasi:", iteration, "| Akurasi:", accuracy, "| K terbaik:", best_k)

    return best_k


# Contoh penggunaan dengan dataset iris
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k_best = knn_gwo(X_train, y_train, X_test, y_test, gwo_iterations=10)
print("Nilai K terbaik:", k_best)
