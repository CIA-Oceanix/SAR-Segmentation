import matplotlib.pyplot as plt


class Plot():
    def __init__(self):
        self.maxQs = []
        self.losss = []
        self.frames = []
        self.rewards = []
        self.survivals = []
        self.expected = []
        self.actions = []

    def Plot(self, history, t):
        plt.ioff()
        plt.clf()
        self.maxQs.append(history['Qsa_maxmean'])
        self.losss.append(history['loss_mean'])
        self.rewards.append(history['cumulative_reward'])
        self.survivals.append(history['frames'])
        self.expected.append(history['exp'])
        self.actions.append(history['actions'])
        self.frames.append(t)

        # plot maxQ
        plt.subplot(3, 2, 1)
        plt.plot(self.frames, self.maxQs)
        plt.xlabel('frame')
        plt.ylabel('maxQ')

        # plot loss
        plt.subplot(3, 2, 2)
        plt.yscale('log')
        plt.plot(self.frames, self.losss)
        plt.xlabel('frame')
        plt.ylabel('loss')

        # plot rewards
        plt.subplot(3, 2, 3)
        plt.plot(self.frames, self.rewards)
        plt.xlabel('frame')
        plt.ylabel('mean reward')

        # plot rewards
        plt.subplot(3, 2, 4)
        plt.plot(self.frames, self.survivals)
        plt.xlabel('frame')
        plt.ylabel('mean survival')

        # plot expected reward
        plt.subplot(3, 2, 5)
        plt.plot(self.frames, self.expected)
        plt.xlabel('frame')
        plt.ylabel('expected reward')

        # plot actions
        plt.subplot(3, 2, 5)
        plt.plot(self.frames, self.actions)
        plt.xlabel('frame')
        plt.ylabel('actions')

        plt.tight_layout()
        plt.savefig('plot.png')

        plt.draw()
