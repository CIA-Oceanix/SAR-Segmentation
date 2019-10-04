import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import shuffle
from datetime import datetime
from shutil import copyfile, move
from glob import glob
import colorsys

import os

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os.path import join, exists, split as splitF
from os import listdir, walk, makedirs

from keras.preprocessing import image as image_utils
from keras import optimizers, regularizers
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow as tf
from bct import modularity_und as SortConfusionMatrix
from keras import backend as K

K.image_dim_ordering()

###################### Hyperparameters ######################
#                    Mode paramaters

names = ['notComic']  # name of the folder hosting the dataset

modes = ['train']  # train, test, testH, trainH, move, export, plotConfuse, makeClusters, plotDiff
modelTypes = [1]  # 1 flat, 2 folder based*
modelIndexs = [0]

currentEpoch = 0  # Not 0 we want to resume a training

#                     Cluster parameters
plot = True
clustering = False  # automatic, import or False

miniClusterSize = 3  # If not 0, communityGamma is an iterable, float else
communityGamma = [x / 10 for x in range(20, 0, -1)]
# communityGamma = 0.8

#                    Dataset parameters

trainNumPic = (0.8, 'absolute')  # each, all, % or absolute
# trainNumPic = (800, 'each') # each, all, or %

testNumPic = (0.2, '%')  # Number of picture per label on which test

validationSplit = 0.1  # Percentage of picture used for validation during train
picSize = (150, 150, 3)
grayscale = picSize[-1] == 1

#                    Network parameters

denseSize = (512, 512)  # Size of the last hidden layers

#                    Training parameters

# Data augmentation
minAugmentation = 25000
params = {'rotation_range': 30, 'width_shift_range': 0.2,
          'height_shift_range': 0.2, 'horizontal_flip': True,
          'zoom_range': (0.75, 1.25), 'fill_mode': 'wrap'}

activation = 'selu'
epochs = 300  # maximal number of epochs
batchSize = 32  # number of picture put in the network with each batch

learningRate = 1 * 10 ** -3
momentum = 0.5
lrDecay = 10 ** -5  # learning rate decay (not weight decay)
weightDecay = 4 * 10 ** -5  # weight decay

dropOut = 0.5
noise = 0.01  # Noise applied after the convolutional layers
threshold = 1  # Not used ATM

#                    Callbacks parameters

earlyStopPatience = 8  # Stop if the loss doesn't decrease since n epochs
ReduceLRPatience = 4  # Decrease the learning rate if loss doesn't decreance since n epochs
ReduceLRFactor = 0.5  # Decrease the learning rate by multiply it by n
ReduceLRCooldown = 1

##################################################################
for name in names:
    if not exists(join('..', 'models')):
        makedirs(join('..', 'models'))
    if not exists(join('..', 'models', name)):
        makedirs(join('..', 'models', name))


class PlotLearning(Callback):
    """Callback generating a fitness plot in a file after each epoch"""

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="Training")
        ax1.plot(self.x, self.val_losses, label="Validation")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Crossentropy')
        ax1.legend()

        ax2.plot(self.x, self.acc, label="Training")
        ax2.plot(self.x, self.val_acc, label="Validation")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('plot.png')


class Model():
    def __init__(self, labels, name, files, folder, mode, index):
        """Initialization of the model

        Input:
        labels -- a list of str
        name -- a string
        files -- a list of list of str
            files[i] -- the list of namefile for labels[i]
        folder -- a string
        """
        self.labels = labels
        self.mode = mode
        self.files = files
        if 'white' in name:
            self.white = True
        else:
            self.white = False
        self.name = name.replace('(white)', '')
        self.folder = folder.replace('(white)', '')
        self.index = index

        for i, files in enumerate(self.files):
            self.files[i] = [[file[-10:-4], file] for file in self.files[i]]
            self.files[i] = sorted(self.files[i])
            self.files[i] = [file[1] for file in self.files[i]]
            # self.files[i] = [file[::-1] for file in self.files[i]]

        self.flatFiles = FlattenList(self.files)
        print()
        print(self.name, ':', ', '.join(self.labels))

    #        print('labels:', self.labels)
    #        print('files:', self.files)

    def ExportSummary(self):
        """Write the model in a .txt"""
        old = sys.stdout
        model = self.ImportModel(False)
        with open('currentModel.txt', 'w') as file:
            sys.stdout = file
            model.summary()
            sys.stdout = old
        del model

    def PrepareData(self):
        """Prepare the input and output of the model

        Output:
        output -- 2d array of int (1 or 2)
            output[i] -- an array with 1 if label[i] correspond to the file, 0 else
        input -- an array containing the images"""
        print()
        print()
        if trainNumPic[1] == 'all':
            self.files = [files[:trainNumPic[0] // len(self.files)] for files in self.files]
        elif trainNumPic[1] == 'each':
            self.files = [files[:trainNumPic[0]] for files in self.files]
        elif trainNumPic[1] == '%':
            total = int(len(self.flatFiles) * trainNumPic[0])
            self.files = [files[:total // len(self.files)] for files in self.files]
        elif trainNumPic[1] == 'absolute':
            total = len(self.flatFiles)

        self.flatFiles = FlattenList(self.files)
        output = [[int(file in files) for file in self.flatFiles] for files in self.files]
        input_ = []
        begin = datetime.now()
        for i, file in enumerate(self.flatFiles):
            eta = ((datetime.now() - begin) / (i + 1) * len(self.flatFiles) + begin).strftime('%H:%M')
            Progress(str(i) + '/' + str(len(self.flatFiles)) + ' - ' + eta)

            x = PrepareImage(file)
            input_.append(x[0])
        print()
        output = np.transpose(np.array(output))
        input_ = np.array(input_)
        c = list(range(len(output)))
        shuffle(c)
        self.output = np.array([output[i] for i in c])
        self.input = np.array([input_[i] for i in c])

    def Train(self, resume=0):
        """Launch the training of the model

        Input:
        resume -- an int
            change the value to continue a previously stopped training
        """
        print('Train', self.name)
        print('labels:', ', '.join(self.labels))
        begin = datetime.now()
        if resume:
            weight = join('..', "models", self.folder, self.name + ".h5")
            model = load_model(weight)
        else:
            model = self.ImportModel(False)
            if self.index != 0:
                print('load from 0')
                model.load_weights(join('..', "models", self.folder, self.name[:-len(str(self.index))] + '0.h5'),
                                   by_name=True)
        calls = [ModelCheckpoint(join('..', "models", self.folder, self.name + ".h5"), save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=earlyStopPatience),
                 ReduceLROnPlateau(monitor='val_loss', factor=ReduceLRFactor,
                                   patience=ReduceLRPatience, cooldown=ReduceLRCooldown),
                 PlotLearning()]

        if self.input.shape[0] < minAugmentation:
            lenValidation = int(self.input.shape[0] * (1 - validationSplit))
            trainingInput = self.input[:lenValidation]
            trainingOutput = self.output[:lenValidation]
            validationInput = self.input[lenValidation:]
            validationOutput = self.output[lenValidation:]

            trainingGenerator = ImageDataGenerator(params).flow(trainingInput, y=trainingOutput,
                                                                batch_size=batchSize)
            validationGenerator = ImageDataGenerator().flow(validationInput, y=validationOutput)

            model.fit_generator(trainingGenerator,
                                callbacks=calls, initial_epoch=resume, epochs=epochs,
                                verbose=1,
                                validation_data=validationGenerator)
        else:
            model.fit(epochs=epochs, verbose=1, validation_split=validationSplit, x=self.input, y=self.output,
                      batch_size=batchSize, callbacks=calls, initial_epoch=resume)
        print('time needed:', datetime.now() - begin)

    def Move(self):
        """Predict the label of files and move them in the corresponding folder"""
        restriction = input('Only keep incorrect prediction ? (y/n): ') == 'y'
        for label in self.labels:
            if not exists(join('..', 'result', label)):
                makedirs(join('..', 'result', label))
        self.LoadModel()

        files = []
        for root, folders, cfiles in os.walk(join('..', 'todo')):
            for file in cfiles:
                if file.endswith('.jpg'):
                    files.append(join(root, file))
        limit = len(files)
        begin = datetime.now()
        for k, file in enumerate(files):
            r = self.Recognize(file)
            labelMax, p = getMaxTuple(r)
            if (restriction and labelMax not in file) or not restriction:
                copyfile(file, join('..', 'result', labelMax, str(k) + '.jpg'))
            ending = (datetime.now() - begin) / (k + 1) * limit + begin
            Progress(str((k + 1)) + '\\' + str(limit) + " | " + ending.strftime('%H:%M'))

    def MoveMode2(self, models):
        """Predict the label of files and move them in the corresponding folder
        Works for model 2

        Input:
        self -- the model with mode 1
        models -- a list of models with mode 2
        """

        hierarch = HierarchIni(models)
        for model in models:
            model.LoadModel()
        for label in self.labels:
            if not exists(join('..', 'result', label)):
                makedirs(join('..', 'result', label))

        files = listdir('todo')
        limit = len(files)
        begin = datetime.now()
        for k, file in enumerate(files):
            labelMax = MaxLabelTree(models[0], hierarch, file)
            copyfile(join('..', 'todo', file), join('..', 'result', labelMax, file))
            ending = (datetime.now() - begin) / (k + 1) * limit + begin
            Progress(str((k + 1)) + '\\' + str(limit) + " | " + ending.strftime('%H:%M'))

    def TestModel(self):
        """Launch the model for tests and compute plots and stats"""
        k = 1
        res = {}
        confuse = [[0 for x in self.labels] for y in self.labels]
        begin = datetime.now()
        undecided = 0
        for files, label in zip(self.files, self.labels):
            if testNumPic[1] == '%':
                nbPerLabel = int(testNumPic[0] * len(files))
                nbTotal = int(testNumPic[0] * len(self.flatFiles))
            elif testNumPic[1] == 'each':
                nbPerLabel = testNumPic[0]
                nbTotal = testNumPic[0] * len(self.files)
            elif testNumPic[1] == 'all':
                nbPerLabel = testNumPic[0] // len(self.files)
                nbTotal = testNumPic[0]

            decided = 0
            if len(files) != 0:
                res[label] = 0
            for i, file in enumerate(files[::-1]):  # Reverse makes more probable to not use the training picts
                if i >= nbPerLabel:
                    continue
                try:
                    r = self.Recognize(file)
                    labelMax, p = getMaxTuple(r)
                    if threshold < p:  # We don't decide
                        undecided += 1
                    else:  # We decide something
                        decided += 1
                        confuse[self.labels.index(label)][self.labels.index(labelMax)] += 1 / nbPerLabel
                        if label == labelMax:  # Our prediction is correct
                            res[label] += 1 / nbPerLabel
                    ending = (datetime.now() - begin) / k * nbTotal + begin
                    Progress(str(k) + '\\' + str(nbTotal) + " | " + ending.strftime('%H:%M'))
                    k += 1
                except Exception as e:
                    print(e)
                    k += 1

        print()
        return res, confuse, undecided / nbTotal

    def Test(self):
        """Launch the model for tests and print plots and stats"""
        self.LoadModel()
        res, confuse, undecided = self.TestModel()
        for label, success in res.items():
            print(label, ':', success)

        print('undecided:', undecided)
        self.PlotConfuse(confuse)

    def PlotConfuse(self, confuse, show=True, dump=True):
        """Plot the confusion matrix and cluster the labels

        Input:
        confuse -- a 2d-matrix of float
            confuse[i][j] -- %age of label[j] being decided as label[i]"""
        if dump:
            name = join('confuse', str(self.mode) + self.name)
            if self.white == True:
                name += '(white)'

            pickle.dump(confuse, open(name + ".p", "wb"))
            pickle.dump(self.labels, open(name + "_header.p", "wb"))
        res = np.mean([confuse[i][i] for i in range(len(confuse))])
        labels = np.array(self.labels)

        # Plot the matrix
        fig, ax = plt.subplots(1, 1)
        p = False

        if clustering == 'automatic':
            # Sort the confusion matrix to make clusters
            # 1/ Get the index

            if miniClusterSize:
                for gamma in communityGamma:
                    ci = SortConfusionMatrix(confuse, gamma=gamma)[0]
                    axes = []

                    c = [(ci[i], i) for i in range(len(ci))]
                    c.sort()
                    p = [c[i][1] for i in range(len(c))]

                    for i in range(1, len(c)):
                        if c[i][0] != c[i - 1][0]:
                            axes.append(i)
                    if len(axes) == 0:
                        continue
                    if len(self.labels) / len(axes) > miniClusterSize:
                        break
            else:
                axes = []
                ci = SortConfusionMatrix(confuse, gamma=communityGamma)[0]

                c = [(ci[i], i) for i in range(len(ci))]
                c.sort()
                p = [c[i][1] for i in range(len(c))]
                for i in range(1, len(c)):
                    if c[i][0] != c[i - 1][0]:
                        axes.append(i)
        elif clustering == 'import':
            clusters = self.ImportClusters()
            print(clusters)
            p = [self.labels.index(label) for label in FlattenList(clusters)]
            axes = []
            for cluster in clusters:
                if not axes:
                    axes.append(len(cluster))
                else:
                    axes.append(axes[-1] + len(cluster))
        if clustering and p:
            for i in range(len(self.labels)):
                if i not in p:
                    print(self.labels[i])
                    p.append(i)
                    axes.append(axes[-1] + 1)

            # 2/ Actually sort
            confuse = np.array(confuse)[p][:, p]
            labels = np.array(self.labels)[p]

            clusters = [[]]
            for i, label in enumerate(labels):
                if i in axes:
                    clusters.append([label])
                else:
                    clusters[-1].append(label)

            # 3/ Plot the frontier between clusters
            inClusterP = 0
            for i, maxAxe in enumerate(axes):
                ax.axhline(maxAxe - 0.5, color=(.1, .1, .1))
                ax.axvline(maxAxe - 0.5, color=(.1, .1, .1))
                if i == 0:
                    minAxe = 0
                else:
                    minAxe = axes[i - 1]
                for x in range(minAxe, maxAxe):
                    for y in range(minAxe, maxAxe):
                        inClusterP += confuse[x, y]
            print('In cluster probability:', inClusterP / len(self.labels))
        else:
            clusters = []
        print('overall succes:', res)
        if np.min(confuse) < 0:
            vmin = -.3
            vmax = .3
        else:
            vmin = 0
            vmax = 1
        img = ax.imshow(confuse, cmap='viridis', vmax=vmax, vmin=vmin, interpolation='nearest')
        fig.colorbar(img, ax=ax)

        # Write the labels  and titles
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(list(range(len(labels))))
        ax.set_yticks(list(range(len(labels))))
        plt.xlabel('Predicted')
        plt.ylabel('Reality')
        plt.title(self.name)
        plt.tight_layout()
        if plot and show:
            plt.show()
        return clusters

    def LoadModel(self):
        """Load the model with weigth"""
        weight = join('..', "models", self.folder, self.name + ".h5")
        self.model = self.ImportModel(weight)

    def Recognize(self, file):
        """Launch the model on one image

        Input:
        file -- a string, the namefile of the image

        Output:
            res -- a list of tuple
                res[i] -- a tuple (str, float) where str is a label"""
        x = PrepareImage(file)
        preds = self.model.predict(x)
        preds = preds[0]
        res = []
        for i in range(len(self.labels)):
            res.append((self.labels[i], preds[i]))
        return res

    def ImportModel(self, weight, input_shape=picSize):
        """Create a model

        Input:
        weight -- False or a string
            False: the model is loaded without weight
            string: the filename of the model, which will be uesd to load the weigth
        input_shape -- (int, int, int), the dimension of one image

        Output:
        modelD - a keras model"""
        modelD = Sequential()
        # Block 1
        modelD.add(Conv2D(64, (3, 3), input_shape=input_shape, activation=activation, padding='same', name='B1C1'))
        # modelD.add(Conv2D(64, (3, 3), activation=activation, padding='same', name='B1C2'))
        modelD.add(MaxPooling2D((2, 2), strides=(2, 2), name='B1P'))

        # Block 2
        modelD.add(Conv2D(128, (3, 3), activation=activation, padding='same', name='B2C1'))
        # modelD.add(Conv2D(128, (3, 3), activation=activation, padding='same', name='B2C3'))
        modelD.add(MaxPooling2D((2, 2), strides=(2, 2), name='B2P'))

        # Block 3
        modelD.add(Conv2D(128, (3, 3), activation=activation, padding='same', name='B3C1'))
        modelD.add(Conv2D(128, (3, 3), activation=activation, padding='same', name='B3C2'))
        # modelD.add(Conv2D(256, (3, 3), activation=activation, padding='same', name='B3C3'))
        modelD.add(MaxPooling2D((2, 2), strides=(2, 2), name='B3P'))
        # Block 4
        modelD.add(Conv2D(128, (3, 3), activation=activation, padding='same', name='B4C1'))
        modelD.add(Conv2D(128, (3, 3), activation=activation, padding='same', name='B4C2'))
        # modelD.add(Conv2D(512, (3, 3), activation=activation, padding='same', name='B4C3'))
        modelD.add(MaxPooling2D((2, 2), strides=(2, 2), name='B4P'))

        # Block 5
        modelD.add(Conv2D(128, (3, 3), activation=activation, padding='same', name='B5C1'))
        modelD.add(Conv2D(128, (3, 3), activation=activation, padding='same', name='B5C2'))
        # modelD.add(Conv2D(512, (3, 3), activation=activation, padding='same', name='B5C3'))
        modelD.add(MaxPooling2D((2, 2), strides=(2, 2), name='B5P'))
        modelD.add(GaussianNoise(noise, name='gaussianNoise'))

        # Classification block
        modelD.add(Flatten(name='flatten'))
        for nbNodes in denseSize:
            modelD.add(Dense(nbNodes, kernel_regularizer=regularizers.l2(weightDecay), activation=activation))
            modelD.add(Dropout(dropOut))
        modelD.add(Dense(len(self.labels), activation='softmax', name='predictions' + str(len(self.labels))))

        sgd = optimizers.SGD(lr=learningRate, momentum=momentum, decay=lrDecay, nesterov=True)

        #        if self.index != 0:
        #            for layer in modelD.layers[:16]:
        #                layer.trainable = False

        modelD.compile(optimizer=sgd,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        if weight:
            print('load', self.name, weight)
            modelD.load_weights(weight)

        return modelD

    def TestHierarch(self, models, mode):
        """Test the architecture of models

        Input:
        self -- the model with mode 1
        models -- a list of models with mode 2 or 3
        mode -- a int

        output
        ress -- a float, the percentage of success of the architecture"""
        confuse = [[0 for x in self.labels] for y in self.labels]
        print(self.labels, '\n')
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for model in models:
                model.LoadModel()
            if mode in (2, 4):
                hierarch = HierarchIni(models)
                if mode == 4:
                    self.LoadModel()

                    invHier = {}
                    for label1 in self.labels:
                        for model1 in models:
                            if label1 in model1.labels:
                                for label, model2 in hierarch.items():
                                    if model1 == model2:
                                        invHier[label1] = label

                begin = datetime.now()
                total = {label: 0 for label in self.labels}

                if testNumPic[1] == '%':
                    nbTotal = int(testNumPic[0] * len(self.flatFiles))
                    nbEach = nbTotal // len(self.files)
                elif testNumPic[1] == 'each':
                    nbTotal = testNumPic[0] * len(self.files)
                    nbTotal = testNumPic[0]
                elif testNumPic[1] == 'all':
                    nbTotal = testNumPic[0]
                    nbEach = nbTotal // len(self.files)
                for j, files in enumerate(self.files):
                    for i, file in enumerate(files[::-1]):
                        if i >= nbEach:
                            continue
                        if mode == 2:
                            labelmax = MaxLabelTree(models[0], hierarch, file, mode)
                        elif mode == 4:
                            labelmax = MaxLabelTree(self, hierarch, file, mode, models=models, invHier=invHier)
                        total[labelmax] += 1
                        confuse[self.labels.index(self.labels[j])][self.labels.index(labelmax)] += 1 / nbEach
                        eta = ((datetime.now() - begin) / (nbEach * j + i + 1) * (nbTotal) + begin).strftime('%H:%M')
                        Progress(str(nbEach * j + i + 1) + '/' + str(nbTotal) + ' | ' + eta)

            #            if mode == 3:
            #                fileLabelModel = SortedDict()
            #                for model in models:
            #                    for file in model.files[0]:
            #                        fileLabelModel[file]=[model, model.labels[0]]
            #                begin = datetime.now()
            #                for i, file in enumerate(fileLabelModel):
            #                    model, label = fileLabelModel[file]
            #                    labelmax = MaxLabelMultiBin(models, file)
            #                    print(file, 'detected as', labelmax)
            #                    confuse[self.labels.index(label)][self.labels.index(labelmax)] += 1/len(model.files[0])
            #                    eta = ((datetime.now()-begin)/(i+1)*len(fileLabelModel)+begin).strftime('%H:%M')
            #                    Progress(str(i+1)+'/'+str(len(fileLabelModel))+' | '+eta)

            print('--------------\nmode:', mode)
            res = 0
            for i in range(len(confuse)):
                print(self.labels[i], ':', confuse[i][i])
                res += confuse[i][i] / len(confuse)
            print(self.labels)
            if self.white:
                pickle.dump(confuse, open(join('confuse', 'H' + self.name + ".p"), "wb"))
            else:
                pickle.dump(confuse, open(join('confuse', 'H' + self.name + "(white).p"), "wb"))
            print(mode, 'succes:', res)
            self.PlotConfuse(confuse)
        return res

    def MakeClusters(self, clusters):
        folders = [join(SplitPath(files[0])[:-1]) for files in self.files]
        for i in range(len(clusters)):
            folder = join(SplitPath(folders[0])[:-1] + [str(i)])
            if not exists(folder):
                makedirs(folder)
        for folder in folders:
            label = SplitPath(folder)[-1]
            for i, cluster in enumerate(clusters):
                if label in cluster:
                    newFolder = join(SplitPath(folder)[:-1] + [str(i)] + [label])
            move(folder, newFolder)

    def ImportClusters(self):
        clusters = {}
        for file in self.flatFiles:
            mother, child = SplitPath(file)[-3:-1]
            if mother not in clusters:
                clusters[mother] = [child]
            elif child not in clusters[mother]:
                clusters[mother].append(child)
        if self.folder != 'illustration123':  # There is no rootted label
            clusters = [clusters[key] for key in clusters if key != self.folder]
        else:
            clusters = [clusters[key] for key in clusters if key != self.folder] + \
                       [[label] for label in clusters[self.folder] if self.folder != 'illustrations123']
        return clusters


def HierarchIni(models):
    """Generate the architecture of the labels

    Input:
    models -- a list of models

    Output:
    hierarch -- a dictionary {k:v}
        k -- a string: the label
        v -- the model used to decide on this label"""
    hierarch = {}
    folder = models[0].folder
    if models[0].white:
        folder += '(white)'
    currentFlats = [(set(glob(join(root, '**', '*.jpg'), recursive=True)), root) for root, subdirs, files in
                    walk(join('.', 'imgs', folder))]
    for model in models:
        modelflat = set(model.flatFiles)
        for currentflat, root in currentFlats:
            if currentflat == modelflat:
                hierarch[SplitPath(root)[-1]] = model
    return hierarch


def MaxLabelTree(model1, hierarch, file, mode, models=None, invHier=None):
    """Decide of a final label with a hierarch architecture of mode 2

    Input:
    model1 -- a model of mode 1
    hiearch -- a dictionary {k:v}
        k -- a string: the label
        v -- the model used to decide on this label
    file -- a string: the filename of a picture

    Output:
    labelmax -- a string"""
    r = model1.Recognize(file)
    labelmax = getMaxTuple(r)[0]
    if mode == 4:
        labelmax = invHier[labelmax]
        if labelmax == name:
            return getMaxTuple(r)[0]
    while labelmax in hierarch:
        r = hierarch[labelmax].Recognize(file)
        labelmax = getMaxTuple(r)[0]
    return labelmax


def MaxLabelMultiBin(models, file):
    """Decide of a final label with a binary architeture of mode 3

    Input:
    models -- a list of models
    file -- a string: the filename of a picture

    Output:
    labelmax -- a string"""
    r = []
    for model in models:
        r.append(model.Recognize(file)[0])
    labelmax, p = getMaxTuple(r)
    return labelmax


def getMaxTuple(r):
    """Return the tuple with the maximal second member

    Input:
    r -- a list of tuple (l, p)
        l -- a str: a label
        p -- a float (between 0 and 1)"""
    pmax = 0
    for label, p in r:
        if p > pmax:
            labelmax = label
            pmax = p
    return labelmax, p


def FlattenList(l):
    """From a list of list return a list flattened

    Input:
    l -- a list of list

    Output:
    flatL -- a list"""
    flatL = []
    for ele in l:
        flatL += ele
    return flatL


def Progress(s):
    """Print a carriage return then a string

    Input:
    s -- a string"""
    sys.stdout.write('\r')
    sys.stdout.write(s + '           ')
    sys.stdout.flush()


def PrepareImage(file):
    """Return a matrix that can go in the network

    Input:
    file -- a string: the namefile of a picture

    Output:
    x -- a 3d array"""
    img = image_utils.load_img(file, target_size=picSize[:2], grayscale=grayscale,
                               interpolation='bicubic')
    x = image_utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def GetColor(z, mini=40, maxi=100):
    """From a float between 0 and 100, return a color

    Input:
    z -- float between 0 and 100
    mini -- the maximum value for red
    maxi -- the minimum value for green

    Output:
    rgb -- a color"""
    if z < mini:
        ratio = 0
    elif z > maxi:
        ratio = 1
    else:
        ratio = (z - mini) / (maxi - mini)
    hue = ratio * 1.2 / 3.6
    rgb = colorsys.hls_to_rgb(hue, 0.45, 1)
    return rgb


def PlotHist(r, s):
    """Plot a histogram

    Input:
    r -- a list of tuple (l, p)
        l -- a string, a label
        p -- a float between 0 and 1, the %age of success
    s -- a string: the title"""
    x, y = [], []
    for label, p in sorted(list(r.items())[:25], key=lambda x: x[1]):
        x.append(label)
        y.append(int(p * 1000) / 10)
    print('Overall probabilities (over tags):', sum(y) / len(y), '%')

    colors = [GetColor(y_) for y_ in y]
    y_pos = np.arange(len(y))
    fig, ax = plt.subplots((1))
    ax.barh(y_pos, y, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(x)
    plt.ylabel('Labels')
    plt.xlabel('Percentage of success')
    plt.title(s)
    plt.show()


def ModelsGenerator(folder, mode, index=None):
    """Generate a list of models with architecture

    Input:
    folder -- a string: the root of the imgs
    mode -- a int between 1 and 3
        mode=1 is a classic classifier
        mode=2 is a tree classifier
        mode=3 is a binary multiclass classifier

    Output:
    models -- a list of models"""
    models = []
    begin = datetime.now()
    allFiles = glob(join('..', 'imgs', folder, '**', '*.jpg'), recursive=True)
    if mode in [1, 3]:
        labels = sorted(list(set([SplitPath(path)[-2] for path in allFiles])))

        filesL = [[] for label in labels]
        label2index = {label: labels.index(label) for label in labels}
        for file in allFiles:
            for label in labels:
                if label == SplitPath(file)[-2]:
                    filesL[label2index[label]].append(file)

        if mode == 1:
            models = [Model(labels, folder, filesL, folder, mode, 0)]
    #        elif mode == 3:
    #            flatFiles = FlattenList(filesL)
    #            for label, files in zip(labels, filesL):
    #                binFiles = [files, [file for file in flatFiles if not file in files]]
    #                random.shuffle(binFiles[1])
    #                binLabels = [label, 'not_'+label]
    #                models.append(Model(binLabels, label, binFiles, folder))
    elif mode in (2, 4):
        archFolders = {}
        archFiles = {}
        for root, dirs, files in walk(join('..', 'imgs', folder)):
            archFolders[root] = [join(root, folder) for folder in dirs]
            archFiles[root] = [join(root, file) for file in files]
        j = 0
        tuples = [(root, dirs) for root, dirs in archFolders.items() if dirs]
        tuples = sorted(tuples)
        models = [None for i in range(len(tuples))]
        if index != None:
            tuples = [tuples[index]]
            j = index
        for root, dirs in tuples:
            modelName = folder + str(j)
            labels = archFolders[root]
            filesL = [glob(join(label, '**', '*.jpg'), recursive=True) for label in labels]

            labels = [SplitPath(label)[-1] for label in labels]
            models[j] = Model(labels, modelName, filesL, folder, mode, j)
            j += 1
    print(datetime.now() - begin, 'to generate models')
    return models


def Launcher(name, modelType, mode, modelIndex):
    """
    Input:
    name -- a string: the root of the imgs
    modelType -- a int between 1 and 3
        modelType=1 is a classic classifier
        modelType=2 is a tree classifier
        modelType=3 is a binary multiclass classifier
    mode -- a string
        'test' -- test a model
        'train' -- train a model
        'trainH' -- train architectured models
        'testH' -- train architectured models
        'move' -- run a model on non-categorized imgs and move the according to the predictions
        "export' -- create .txt and a .png describing the model
        'plotConfuse' -- show the confusion matrix
        'makeClusters' -- divide the folders into clusters"""
    if mode in ['testH', 'trainH']:
        models = ModelsGenerator(name, modelType)
    else:
        models = ModelsGenerator(name, modelType, index=modelIndex)
    if mode == 'test':
        if modelIndex == 0 and modelType == 4:
            ModelsGenerator(name, 1).Test()
        else:
            models[modelIndex].Test()
    elif mode == 'train':
        models[modelIndex].PrepareData()
        models[modelIndex].Train(resume=currentEpoch)
    elif mode == 'trainH':
        for model in models:
            with tf.Session() as sess:
                model.PrepareData()
                model.Train()
            sess
    elif mode == 'testH':
        ModelsGenerator(name, 1)[0].TestHierarch(models, modelType)
    elif mode == 'move':
        if modelType == 1:
            models[0].Move()
        else:
            ModelsGenerator(name, 1)[0].MoveMode2(models)
    elif mode == 'export':
        models[modelIndex].ExportSummary()
    elif mode == 'plotConfuse':
        confuse = pickle.load(
            open(join('confuse', str(models[modelIndex].mode) + models[modelIndex].name + ".p"), "rb"))
        if modelType == 4:
            model = ModelsGenerator(name, 1)[0]
            model.PlotConfuse(confuse, dump=False)
        else:
            models[modelIndex].PlotConfuse(confuse, dump=False)
    elif mode == 'makeClusters':
        confuse = pickle.load(
            open(join('confuse', str(models[modelIndex].mode) + models[modelIndex].name + ".p"), "rb"))
        clusters = models[modelIndex].PlotConfuse(confuse, show=False, dump=False)
        models[modelIndex].MakeClusters(clusters)
    elif mode == 'plotDiff':
        confuse1 = np.array(pickle.load(open(join('confuse', '1illustrations.p'), "rb")))
        confuse2 = np.array(pickle.load(open(join('confuse', 'Hillustrations.p'), "rb")))
        confuse = confuse2 - confuse1
        models[modelIndex].PlotConfuse(confuse, dump=False)
        models[modelIndex].PlotConfuse(confuse1, dump=False)
        models[modelIndex].PlotConfuse(confuse2, dump=False)
    else:
        print('incorrect command:', mode)


def SplitPath(path):
    head, tail = splitF(path)
    if not head:
        return [tail]
    else:
        return SplitPath(head) + [tail]


if __name__ == '__main__':
    print('BEGIN:', datetime.now().strftime('%H:%M'))
    for name, modelType, mode, modelIndex in zip(names, modelTypes, modes, modelIndexs):
        with tf.Session() as sess:
            Launcher(name, modelType, mode, modelIndex)
        sess
    print('FINISH', datetime.now().strftime('%H:%M'))
