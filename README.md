# Dysarthria Speech Recognition

#### Capturing every voice with care and clarity

## Overview

Dysarthria is a condition that makes it hard for some people to speak clearly. It happens because of problems with the brain and nervous system. More than a million people around the world have dysarthria. People with dysarthria might slur their words, talk very slowly, strain their voice, or have trouble controlling how loud or soft they speak. This makes it difficult for them to communicate with others. Having a hard time communicating can make people with dysarthria feel upset, frustrated, or isolated from others. Not many people know about dysarthria or how to help those who have it. However, new technologies like voice recognition software that can understand different types of speech may help. These could allow faster and more accurate communication between people with dysarthria and those they are speaking with. This would make it easier for their message to come across clearly.

## Description

We are building a Automatic Speech Recogntion model for dysarthric patients by fine tuning the Whisper model provided by OpenAI.

The project is structured into four main Jupyter notebooks that sequentially cover the steps required to process audio data, create a dataset, fine tuning a Whisper-based model, and perform inference.

## Notebook description

### 1. Dataset Creation

The notebook shows the steps to create a dataset from prompt text files and corresponding audio (WAV) files. It starts by importing necessary libraries and defining file paths. An empty DataFrame is created to store the dataset.
The code iterates through the prompt and audio file directories, matching each WAV file with its corresponding prompt file based on naming conventions. It extracts relevant information like sex, control status, and microphone type from the file paths.
For each matched pair, the audio is processed using librosa to load the samples and sample rate, while the prompt text is read from the file. A new row is created combining this information and appended to the dataset DataFrame.
Finally, summary statistics are printed, the mapping between WAV and prompt files is saved as JSON, and the complete dataset is saved as a pickle file for further analysis and model training.


### 2. Audio Processing and DataFrame Cleaning

This script prepares a dataset for training an audio transcription model using Whisper. It installs required packages, imports libraries, and sets up Whisper components. It mounts Google Drive to access data, loads the data, and preprocesses it by handling contractions, removing unwanted text, and filtering specific samples. The data is then split into train, validation, and test sets. The processed data is saved and converted to PyTorch datasets. It computes log-Mel features from audio samples and encodes target text using Whisper components. Finally, it maps the preparation function to the datasets and saves them for model training.


### 3. Training

This script sets up the environment, fine-tunes a pre-trained Whisper speech recognition model for English transcription, and evaluates its performance. It installs required packages, imports libraries, loads preprocessed datasets, defines data collation and evaluation metrics, loads the pre-trained Whisper model, configures it for English generation, sets training arguments, initializes the trainer, trains the model, saves the trained model, and finally evaluates it on the test dataset. The main steps involve data preparation, model setup, training configuration, model fine-tuning, and performance evaluation using the Word Error Rate metric.

### 4. Inference

This code demonstrates how to perform speech-to-text transcription inference using a pre-trained Whisper model. It installs necessary dependencies, mounts Google Drive, defines dataset path, and loads the trained Whisper model along with its processor and feature extractor components. It then loads an audio file, extracts log-mel audio features from it using the feature extractor, preprocesses the features, and passes them to the loaded Whisper model for inference to obtain the transcribed text output. Finally, it decodes and prints the transcribed text. The code also shows an alternative approach to directly use a pre-trained Whisper model for inference without any additional fine-tuning.


## Installation

- Ensure all dependencies are installed as listed in each notebook's 'Adding Dependencies' section.
- Data must be accessible either locally or via Google Drive as specified in the notebooks.

- Cleaned data in pickle format can be downloaded from the link given, you can skip the notebook 1 if you start with this: [Link](https://drive.google.com/file/d/1UT2gm0I0XY_7ykfBPFKLljAhRHO59n-z/view?usp=drive_link)


- If you are just want to train, validate and test the model use the links below:
[Train dataset](https://drive.google.com/drive/folders/1VQgAAbvnbCS1CBwGeaduyZQHjztgVG5g?usp=drive_link)
[Validate dataset](https://drive.google.com/drive/folders/1KxaOUe55EgGvTPJ2XfnxbrQk5stIwSTu?usp=drive_link)
[Test dataset](https://drive.google.com/drive/folders/1fxcsi3VE5AjK-cy4MnuwBra3lAMIdJEL?usp=drive_link)


- If you trying the inference, get the fine tuned model from this link: [Link](https://drive.google.com/drive/folders/1il7hyjwOR2nYyJfMdckTcKwCcp6JOKX2?usp=drive_link)

## Usage

Follow the notebooks in the sequence they are presented:
1. Start with `Dataset Creation.ipynb` to structure your dataset.
1. The follow with `audio_processing_dataframe_cleaning.ipynb` for in depth preprocessing.
3. Train the model using `Training.ipynb`.
4. Perform inference with `inference.ipynb`.

## Contributing
Contributions are welcome. Please fork the project and submit a pull request with your suggested changes.

## License
NA
