# DeepLearning

- [Requirements](#requirements)
- [Use](#use)
  - Image2Image
      - [Autoencoders](#autoencoders)
      - [Saliency](#saliency)
  - Categorization
      - [Categorizer (image2tag)](#categorization-on-images)
      - [Speech Recognition (sound2tag)](#speech-recognition-specifics)
  - [Optical Character Recognition](#optical-character-recognition)
  - [Reinforcement Learning](#)
  - GANs
      - [StyleGan](#)
      - [CycleGan](#)
- [Dataset location](#)

## Requirements

- os
- sys
- glob
- tqdm
- numpy
- cv2
- scipy
- PIL
- datetime
- matplotlib
- seaborn
- tensorflow
- keras
- Rignak_Misc

###### Reinforcement learning specifics

- pygame
- skimage

###### Speech recognition specifics

- sounddevice

###### StyleGan specififics

- argparse
- six
- threading
- traceback
- typing
- pprint
- json
- times
- types
- enum
- copy
- io
- pathlib
- pickle
- platform
- re
- shutil
- zipfile
- collections
- tensorboard
- inspect
- uuid


## Use

Autoencoders and categorizer revolve around the *train.py*

````shell
>>> python train.py {model_type} {task} {dataset} {output_prefix}
````

- **{task}** is either "autoencoder" or "categorizer" ;

- **{model_type}** can be "unet" or "flat" for the task "autoencoder" and "flat" for "categorizer" ;

- **{dataset}** is the name of a directory inside the *./datasets* folder. See [dataset location](#) for the specifics of the filesystem architecture.

### Autoencoders

###### Inputs

````shell
>>> python train.py unet autoencoder fav-rignak name=fav-rignak
````

###### Outputs

- **_outputs\summary\fav-rignak.txt**, the summary of the neuron network ;
- **_outputs\models\fav-rignak.h5**, the pickled model ;
- **_outputs\history\fav-rignak.png**, the evolution of the loss during the training on both training and validation sets ;
- **_outputs\example\fav-rignak_current.png**, sample input/output/groundtruth of the model during the training on the validation set.

###### Example (NYA)

![](_outputs/example/fav-rignak.png)

### Saliency

Knowing only a classification of the training set, we try to draw the segmentation.

###### Inputs

````shell
>>> python train.py flat saliency open_eyes name=open_eyes
````

Supports dataset with two and three class.

###### Outputs

- **_outputs\summary\open_eyes.txt**, the summary of the neuron network ;
- **_outputs\models\open_eyes.h5**, the pickled model ;
- **_outputs\history\open_eyes.png**, the evolution of the loss during the training on both training and validation sets ;
- **_outputs\example\open_eyes_current.png**, sample input/output/groundtruth of the model during the training on the validation set.

###### Example

![](_outputs/example/open_eyes.png)
![](_outputs/example/eye_color.png)


### Categorization on images

###### Inputs

````shell
>>> python train.py flat categorizer waifu name=waifu
````

###### Outputs

- **_outputs\summary\waifu.txt**, the summary of the neuron network ;
- **_outputs\models\waifu.h5**, the pickled model ;
- **_outputs\history\waifu.png**, the evolution of the loss during the training on both training and validation sets ;
- **_outputs\confusion\waifu.png**, confusion of the model during the training on the validation set.


###### Example

![](_outputs/confusion/waifu.png)

### Categorization on sound

Using the spectrogram of the sound, we convert the .wav into .png. Thus, we can apply the classic computer vision methods.

###### Dataset examples

- 経済

![](SpeechRecognition/datasets/japanese/keizai/0.png)
![](SpeechRecognition/datasets/japanese/keizai/1.png)
![](SpeechRecognition/datasets/japanese/keizai/2.png)

- 消しゴム

![](SpeechRecognition/datasets/japanese/keshigomu/0.png)
![](SpeechRecognition/datasets/japanese/keshigomu/1.png)
![](SpeechRecognition/datasets/japanese/keshigomu/2.png)


###### Inputs

````shell
>>> cd SpeechRecognition
>>> python train.py japanese
````

###### Outputs

- **_outputs\summary\japanese.txt**, the summary of the neuron network ;
- **_outputs\models\japanese.h5**, the pickled model ;
- **_outputs\history\japanese.png**, the evolution of the loss during the training on both training and validation sets ;
- **_outputs\confusion\japanese.png**, confusion of the model during the training on the validation set.

###### Examples

![](_outputs/confusion/japanese.png)

### Optical Character Recognition