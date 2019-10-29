---
layout: post
title:  "Applying Audio Deep Learning to Micro-Doppler Data"
date:   2019-08-19 10:00:00
comments: false
---

For my Master's project I performed an analysis of a micro-Doppler dataset, in order to see whether it could be fed and processed by an audio classification model and consequently be listened to as sound. This blog post provides a summary of some of my results.

<!--more-->

## What is micro-Doppler?

Recall what you hear when an ambulance with its siren on passes you. As it moves towards you and moves away from you, the pitch of the siren seems to change. This is caused by the **Doppler effect**, which is defined as the observed change in frequency of waves caused by relative motion between the wave source and the observer.

<iframe width="420" height="315" src="https://www.youtube.com/embed/imoxDcn2Sgo" frameborder="0" allowfullscreen></iframe>
*Example of the Doppler effect produced by a siren.*

If the moving target produces a vibration or rotation in addition to its primary movement, this causes additional frequency shifts, referred to as **micro-Doppler shifts**. Due to the intensity of the micro-Doppler effect being dependent on velocity and direction, each movement will possess a distinct micro-Doppler signature, which is useful for activity recognition.

## The Project

The task of activity recognition using micro-Doppler signatures is relatively new and so the amount of labelled data is limited. As sound and Wi-Fi signals (which is a common modality for collecting micro-Doppler data) share similarities, and sound is a medium that has been extensively studied in the machine learning community, this work investigated whether audio deep learning architectures can be leveraged to improve micro-Doppler activity recognition in terms of accuracy and computational time. 

The micro-Doppler dataset I was working with stored the information in the form of spectrograms. [Spectrograms](https://en.wikipedia.org/wiki/Spectrogram) are a visual representation of a spectrum of frequencies of a signal as it varies by time. They are constructed from a 'normal' signal (one that records amplitude changes over time) by performing a transform like a Fourier transform. The audio deep learning model used spectrograms generated from audio, so some adjustments had to be made to the micro-Doppler spectrograms to make it appear more like audio. I experimented with more complex machine learning transformations and standard image processing techniques (since spectrograms can be treated like images).


## Constructed Sound Results

Once suitable transformations were made, the transformed micro-Doppler spectrograms were converted back into the time-amplitude domain and listened to like sound clips. Below I have included results from two movement classes (sitting and standing), generated using different transformations. No resampling was conducted so that the sound quality of the samples is unaffected. Each sample is divided by a sine wave sound to distinguish between the samples. The justification for these transformations are described in my dissertation.

Due to the sampling rate of these files not matching audio, they do not play in Chrome. Please listen in Microsoft Edge, another browser that supports atypical sampling rates or download if the above options do not work.

### Unprocessed Data
The spectrograms in their unprocessed form converted back into the time-amplitude domain.

[Sitting](/assets/sound/unprocessed_sit.wav) [Standing](/assets/sound/unprocessed_stand.wav) 

### DSGAN Transformed Data
This data was transformed using DSGAN trained for 1 epoch with L2 loss.

[Sitting](/assets/sound/ds_sit.wav) [Standing](/assets/sound/ds_stand.wav) 

### DiscoGAN Transformed Data
This data was transformed using DiscoGAN trained for 5 epochs with L2 loss.

[Sitting](/assets/sound/disco_sit.wav) [Standing](/assets/sound/disco_stand.wav) 

### CycleGAN Transformed Data
This data was transformed using CycleGAN trained for 1 epoch.

[Sitting](/assets/sound/cycle_sit.wav) [Standing](/assets/sound/cycle_stand.wav) 

### Z-score Standardised Data
The data was z-score standardised to 0 mean and unit variance.

[Sitting](/assets/sound/normalised_sit.wav) [Standing](/assets/sound/normalised_stand.wav) 

### Standardised and Rescaled Data
After z-score standardisation, the mean and variance was changed to match the sound data's mean and variance.

[Sitting](/assets/sound/scale_sit.wav) [Standing](/assets/sound/scale_stand.wav) 

### Scaled Data using amplitudes across frequency axis
The data was normed and rescaled using the average amplitudes corresponding to each frequency bin.

[Sitting](/assets/sound/freq_sit.wav) [Standing](/assets/sound/freq_stand.wav) 

### Scaled Data using amplitudes across time axis
The data was normed and rescaled using the average amplitudes corresponding to each time bin.

[Sitting](/assets/sound/time_sit.wav) [Standing](/assets/sound/time_stand.wav) 



As you can tell, they are not very distinguishable from each other! This may be because the predominant features for all classes are contained near zero Doppler or perhaps due to the conversion function being inaccurate.

If you would like to listen to more clips for the other classes, you can find the .wav files [here](https://github.com/ktmai/msc-dissertation/tree/master/audio).