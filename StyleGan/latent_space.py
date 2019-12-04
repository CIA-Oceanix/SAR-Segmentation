import matplotlib.pyplot as plt
import numpy as np

from Rignak_DeepLearning import Genetics


def imshow(images, titles):
    plt.figure(figsize=(15, 10))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image)
        if len(titles) > i:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()

def steps_with_direction(generative, latent_vector, direction, coefficients):
    latent_vectors = [latent_vector + coeff * direction for coeff in coefficients]
    images = generative(latent_vectors)
    titles = [f"Coeff: {coeff}" for coeff in coefficients]

    imshow(images, titles)


def transfer_style(generative, latent_vector, style_vector, layers_to_swap):
    titles = ['Style', 'Source']
    latent_vectors = [latent_vector, style_vector]
    for i, layer_index in enumerate(layers_to_swap):
        titles.append(f'{layer_index}/{latent_vector.shape[0]} layers swapped')
        latent_vectors.append(latent_vector.copy())
        latent_vectors[-1][:layer_index] = style_vector[:layer_index]

    images = generative(latent_vectors)

    imshow(images, titles)


def random_steps(generative, latent_vector, step=0.01, n=4, layers=(-1,), loss=None):
    titles = ['Source Image']
    latent_vectors = np.array([latent_vector.copy() for _ in range(n + 1)])
    for i in range(1, n + 1):
        for layer in layers:
            latent_vectors[i][layer] += (np.random.random(latent_vectors[i][layer].shape) - 0.5) * step
        if loss:
            titles.append(loss(latent_vectors[i]))

    images = generative(latent_vectors)
    imshow(images, titles)


def steps_with_genetics(generative, loss, source, destination, adn_shape, population_size, epochs, mutation_step):
    new_vectors = Genetics.fit(source, destination, adn_shape=adn_shape, loss=loss, population_size=population_size,
                               epochs=epochs, mutation_step=mutation_step)
    latent_vectors = [source] + list(new_vectors) + [destination]
    images = generative(latent_vectors)
    imshow(images, titles=[])
