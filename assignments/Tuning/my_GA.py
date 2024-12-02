import numpy as np

class my_GA:
    def __init__(self, param_grid, model, data_X, data_y, decision_boundary, obj_func, generation_size=100, selection_rate=0.5,
                 mutation_rate=0.01, crossval_fold=5, max_generation=100, max_life=3):
        self.param_grid = param_grid
        self.model = model
        self.data_X = data_X
        self.data_y = data_y
        self.decision_keys = list(decision_boundary.keys())
        self.decision_boundary = list(decision_boundary.values())
        self.obj_func = obj_func
        self.generation_size = int(generation_size)
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.crossval_fold = int(crossval_fold)
        self.max_generation = int(max_generation)
        self.max_life = int(max_life)
        self.life = self.max_life
        self.iter = 0
        self.generation = []
        self.pf_best = []
        self.evaluated = {None: -1}
        self.best_params_ = None

    def initialize_population(self):
        population = []
        for _ in range(self.generation_size):
            individual = {param: np.random.choice(values) for param, values in self.param_grid.items()}
            population.append(individual)
        return population

    def evaluate_population(self, population):
        scores = []
        for individual in population:
            model = self.model(**individual)
            score = self.obj_func(model, self.data_X, self.data_y)
            scores.append((score, individual))
        return scores

    def select_parents(self, scores):
        scores.sort(reverse=True, key=lambda x: x[0])
        num_parents = int(self.selection_rate * len(scores))
        return [individual for _, individual in scores[:num_parents]]

    def crossover(self, parents):
        new_population = []
        for _ in range(self.generation_size):
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child = {}
            for param in self.param_grid.keys():
                child[param] = np.random.choice([parent1[param], parent2[param]])
            new_population.append(child)
        return new_population

    def mutate(self, population):
        for individual in population:
            if np.random.rand() < self.mutation_rate:
                param = np.random.choice(list(self.param_grid.keys()))
                individual[param] = np.random.choice(self.param_grid[param])
        return population

    def run(self):
        population = self.initialize_population()
        best_score = -np.inf
        best_individual = None
        life = 0

        for generation in range(self.max_generation):
            scores = self.evaluate_population(population)
            parents = self.select_parents(scores)
            population = self.crossover(parents)
            population = self.mutate(population)

            current_best_score, current_best_individual = max(scores, key=lambda x: x[0])
            if current_best_score > best_score:
                best_score = current_best_score
                best_individual = current_best_individual
                life = 0
            else:
                life += 1

            if life >= self.max_life:
                break

        self.best_params_ = best_individual