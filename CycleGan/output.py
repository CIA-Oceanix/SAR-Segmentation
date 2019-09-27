import matplotlib.pyplot as plt
import os
import numpy as np


def sample_images(model, epoch, batch_i):
    os.makedirs(f'output/{model.dataset_name}', exist_ok=True)

    imgs_A = model.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    imgs_B = model.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

    # Translate images to the other domain
    fake_B = model.g_AB.predict(imgs_A)
    fake_A = model.g_BA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = model.g_BA.predict(fake_B)
    reconstr_B = model.g_AB.predict(fake_A)

    plt.figure(figsize=(20, 6))
    plt.subplot(2, 6, 1)
    plt.imshow(0.5 * imgs_A[0] + 0.5)
    plt.subplot(2, 6, 2)
    plt.imshow(0.5 * fake_B[0] + 0.5)
    plt.subplot(2, 6, 3)
    plt.imshow(0.5 * reconstr_A[0] + 0.5)
    plt.subplot(2, 6, 4)
    plt.imshow(0.5 * np.mean(np.abs(imgs_A - fake_B)[0], axis=-1) + 0.5, cmap='hot')
    plt.subplot(2, 6, 5)
    plt.imshow(0.5 * np.mean(np.abs(fake_B - reconstr_A)[0], axis=-1) + 0.5, cmap='hot')
    plt.subplot(2, 6, 6)
    plt.imshow(0.5 * np.mean(np.abs(imgs_A - reconstr_A)[0], axis=-1) + 0.5, cmap='hot')

    plt.subplot(2, 6, 7)
    plt.imshow(0.5 * imgs_B[0] + 0.5)
    plt.subplot(2, 6, 8)
    plt.imshow(0.5 * fake_A[0] + 0.5)
    plt.subplot(2, 6, 9)
    plt.imshow(0.5 * reconstr_B[0] + 0.5)
    plt.subplot(2, 6, 10)
    plt.imshow(0.5 * np.mean(np.abs(imgs_B - fake_A)[0], axis=-1) + 0.5, cmap='hot')
    plt.subplot(2, 6, 11)
    plt.imshow(0.5 * np.mean(np.abs(fake_A - reconstr_B)[0], axis=-1) + 0.5, cmap='hot')
    plt.subplot(2, 6, 12)
    plt.imshow(0.5 * np.mean(np.abs(imgs_B - reconstr_B)[0], axis=-1) + 0.5, cmap='hot')

    plt.savefig(f"output/{model.dataset_name}/{epoch}_{batch_i}.png")
    plt.savefig(f"output/{model.dataset_name}/current.png")
    plt.close()


def plot_metrics(metrics):
    plt.figure(figsize=(8, 4))
    plt.subplot(131)
    plt.plot(metrics['DA_acc'])
    plt.plot(metrics['DB_acc'])
    plt.ylim(0, 100)
    plt.title('Discriminator Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(('Discriminator A', 'Discriminator B'))

    plt.subplot(132)
    plt.plot(metrics['DA_loss'])
    plt.plot(metrics['DB_loss'])
    plt.semilogy()
    plt.title('Discriminator Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(('Discriminator A', 'Discriminator B'))

    plt.subplot(133)
    for metric in ("G_loss", "G_adv", "G_recon", "G_id"):
        plt.plot(metrics[metric])

    plt.semilogy()
    plt.title('Generator Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(('Loss', 'Adv', 'Recon', 'Id'))
    
    plt.savefig(f"output/_metrics.png")
    plt.close()
