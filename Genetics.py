import numpy as np
import matplotlib.pyplot as plt
import tqdm.auto as tqdm

STEPS = 5
N_LAYERS = 18
LAYER_SHAPE = 512
ADN_SHAPE = (STEPS, N_LAYERS, LAYER_SHAPE)

MUTATION_RATE = 10 ** -3
MUTATION_STEP = 10 ** -1

POPULATION_SIZE = 100
ELITE_SIZE = 10
EPOCHS = 5


def plot_loss(min_losses, mean_losses):
    plt.plot(min_losses)
    plt.plot(mean_losses)
    plt.yscale('log')
    plt.savefig('temp.png')


def euclidian_distance(a, b):
    a = np.array([e for axis in a for e in axis])
    b = np.array([e for axis in b for e in axis])

    return np.linalg.norm(a - b)


def cumulative_distance(path, source, destination, distance_function=euclidian_distance):
    distances = [distance_function(source, path.adn[0])]
    for i_point in range(path.adn.shape[0] - 1):
        distances.append(distance_function(path.adn[i_point], path.adn[i_point + 1]))
    distances.append(distance_function(path.adn[-1], destination))

    return sum([distance for distance in distances])


class Path():
    def __init__(self, source, destination, adn_shape=ADN_SHAPE,
                 mutation_rate=MUTATION_RATE,
                 mutation_step=MUTATION_STEP):
        self.mutation_rate = mutation_rate
        self.mutation_step = mutation_step
        self.fitness = 0

        self.adn = np.zeros(adn_shape)
        if source is not None and destination is not None:
            for i in range(self.adn.shape[0]):
                self.adn[i] = source + (destination - source) * ((i + 1) / (self.adn.shape[0] + 1))
            # self.mutate()
        else:
            self.adn = np.random.random(self.adn.shape)

    def mutate(self):
        mutation = (np.random.random(np.product(self.adn.shape)))
        mutation[mutation < self.mutation_rate] = 1
        mutation[mutation != 1] = 0
        mutation = mutation.reshape(self.adn.shape)
        self.adn += np.multiply(mutation, np.random.random(mutation.shape) - 0.5) * self.mutation_step

        return self

    def birth(self, parent1, parent2):
        choice = (np.random.randint(0, 2, (np.product(self.adn.shape))))
        choice = choice.reshape(self.adn.shape)
        self.adn = np.where(choice, parent1.adn, parent2.adn)

    def get_fitness(self, source, destination, distance_function=euclidian_distance):
        if self.fitness == 0:
            self.fitness = 1 / cumulative_distance(self, source, destination, distance_function=distance_function) ** 2
        return self.fitness


def select(ranked_population, elite_size=ELITE_SIZE):
    selected = [elite for elite, cumulative_fitness in ranked_population[:elite_size]]

    for i in range(len(ranked_population) - elite_size):
        pick = np.random.random() * ranked_population[-1][1]
        for j, (individual, cumulative_fitness) in enumerate(ranked_population):
            if pick < cumulative_fitness:
                selected.append(individual)
                break
    return selected


def breed(population, population_size):
    new_population = []
    for _ in range(population_size):
        parent1 = np.random.choice(population)
        parent2 = np.random.choice(population)
        child = Path(source=None, destination=None, adn_shape=parent1.adn.shape, mutation_rate=parent1.mutation_rate,
                     mutation_step=parent1.mutation_step)
        child.birth(parent1, parent2)
        new_population.append(child)
    return new_population


def rank(population, source, destination, loss=euclidian_distance, pbar=None, bar_text=''):
    fitness_dic = {}
    for i in range(len(population)):
        if pbar is not None:
            pbar.set_description(bar_text)
            pbar.update(1)
        fitness_dic[i] = population[i].get_fitness(source, destination, distance_function=loss)
    cumulative_fitness = 0
    ranked_population = []
    for fitness, rank in sorted([(value, key) for key, value in fitness_dic.items()], reverse=True):
        cumulative_fitness += fitness
        ranked_population.append((population[rank], cumulative_fitness))
    return ranked_population


def next_generation(ranked_population, elite_size=ELITE_SIZE):
    selection = select(ranked_population, elite_size)
    elite_population = selection[:elite_size]
    new_population = breed(selection, len(ranked_population) - len(elite_population)) + elite_population
    new_population = [new_individual.mutate() for new_individual in new_population]
    return new_population


def fit(source, destination, adn_shape=ADN_SHAPE, population_size=POPULATION_SIZE, epochs=EPOCHS, elite_size=ELITE_SIZE,
        loss=euclidian_distance, mutation_step=MUTATION_STEP):
    population = [Path(source, destination, adn_shape=adn_shape, mutation_step=mutation_step)
                  for _ in range(population_size)]
    print('Initial distance:', 1 / population[0].get_fitness(source, destination, loss))
    ranked_population = rank(population, source, destination, loss=loss)

    min_losses = []
    mean_losses = []
    pbar = tqdm.tqdm(range(epochs * len(population)))
    for i in range(epochs):
        min_losses.append(1 / ranked_population[0][1])
        mean_losses.append(1 / np.mean([individual.fitness for individual, _ in ranked_population]))
        ranked_population = rank(population, source, destination, loss=loss,
                                 pbar=pbar,
                                 bar_text=f" - Gen {i + 1}/{epochs} - Min loss: {int(min_losses[-1])}, "
                                 f"Mean loss: {int(mean_losses[-1])}")
        population = next_generation(ranked_population, elite_size=elite_size)
    pbar.close()
    return ranked_population[0][0].adn


def main():
    source = np.zeros((18, 512))
    destination = np.ones((18, 512))
    fit(source, destination, population_size=100, epochs=100, elite_size=10,
        adn_shape=(STEPS, source.shape[0], source.shape[1]))


if __name__ == "__main__":
    main()
