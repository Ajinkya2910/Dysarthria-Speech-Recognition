# Dysarthria Speech Recognition

#### Capturing every voice with care and clarity

## Overview

Dysarthria is a condition that makes it hard for some people to speak clearly. It happens because of problems with the brain and nervous system. More than a million people around the world have dysarthria. People with dysarthria might slur their words, talk very slowly, strain their voice, or have trouble controlling how loud or soft they speak. This makes it difficult for them to communicate with others. Having a hard time communicating can make people with dysarthria feel upset, frustrated, or isolated from others. Not many people know about dysarthria or how to help those who have it. However, new technologies like voice recognition software that can understand different types of speech may help. These could allow faster and more accurate communication between people with dysarthria and those they are speaking with. This would make it easier for their message to come across clearly.

## Description

We are building a Automatic Speech Recogntion model for dysarthric patients by fine tuning the Whisper model provided by OpenAI.

The project is structured into four main Jupyter notebooks that sequentially cover the steps required to process audio data, create a dataset, fine tuning a Whisper-based model, and perform inference.

### 1. Dataset Creation

The details about the steps for dataset creation are provided below:

1. **Mounting Google Drive (if running on Colab)**: If the code is running on Google Colab, this step mounts Google Drive to access files stored there.

2. **Copying Dataset (if running on Colab)**: The code copies the dataset from Google Drive to the Colab runtime environment to expedite data processing.

3. **Importing Libraries**: Necessary libraries such as pandas, tqdm, json, scipy, librosa, and numpy are imported for data manipulation, file handling, and audio processing.

4. **Defining Paths**: Paths for the prompt files and audio files are provided. These paths are either from Google Drive or local directory depending on where the code is executed.

5. **Creating Dataset DataFrame**: An empty DataFrame with predefined columns and data types is created to store information extracted from prompt and WAV files. This DataFrame is then saved as a CSV file.

6. **Iterating through Prompt Files**: The code iterates through each prompt file directory and attempts to access prompt files within each directory.

7. **Iterating through Audio Files**: Similarly, the code iterates through each audio file directory and attempts to access WAV files within each directory.

8. **Matching WAV Files with Prompt Files**: For each WAV file, the code attempts to find a corresponding prompt file based on the file naming convention.

9. **Summary Statistics**: Summary statistics such as the number of WAV files not matched with prompt files, total WAV files, and total prompt files are printed.

10. **Saving Mapping to JSON**: The mapping between WAV files and prompt files is saved as a JSON file for future reference.

11. **Reading JSON File**: The JSON file containing the mapping between WAV files and prompt files is read to retrieve the information.

12. **Processing Audio Files**: Each WAV file is processed using librosa to load audio samples and sample rate. Additional attributes such as sex, control, and micType are extracted from the file path.

13. **Reading Prompt Files**: The content of the corresponding prompt file is read.

14. **Creating Dataset Rows**: A new row is created with information extracted from both the WAV and prompt files.

15. **Appending Rows to Dataset**: Each new row is appended to the dataset DataFrame.

16. **Saving Dataset**: The final dataset is saved as a pickle file for further analysis.

Each step contributes to the creation of a comprehensive dataset containing information about dysarthria speech samples and their corresponding prompts, which can be used for training and evaluation purposes in machine learning models.


### 2. Audio Processing and DataFrame Cleaning

This script does some further preprocessing to prepare dataset for training a model. The steps are:

1. **Dependencies Installation**: Installing necessary packages like `datasets`, `evaluate`, and `jiwer`.
2. **Library Imports**: Importing required libraries such as pandas, numpy, matplotlib, etc.
3. **Whisper Model Components**: Importing and setting up components of the Whisper model like `WhisperFeatureExtractor` and `WhisperTokenizer`.
4. **Mounting Google Drive**: Mounting Google Drive to access and save data.
5. **Loading Data**: Loading data, likely for training, from Google Drive.
6. **Preprocessing**: Preprocessing the loaded data, which includes:
   - Handling contractions and special characters
   - Normalizing text (lowercasing)
   - Removing invalid data (e.g., images links)
   - Handling missing values
   - Filtering specific data (e.g., only keeping samples of Dysarthric patients)
7. **Data Splitting**: Splitting the data into train, validation, and test sets.
8. **Saving Processed Data**: Saving the preprocessed data for later use.
9. **Converting to PyTorch Dataset**: Converting the pandas dataframes to datasets compatible with PyTorch.
10. **Preparing Dataset**: Preparing the dataset by:
    - Computing log-Mel input features from audio samples using `WhisperFeatureExtractor`.
    - Encoding target text to label ids using `WhisperTokenizer`.
11. **Mapping and Saving Dataset**: Mapping the preparation function to the datasets and saving them to disk.

This script sets up the data pipeline, preprocesses the data, and prepares it for training a model for audio transcription tasks using the Whisper model.


### 3. Training

This script sets up the environment, fine-tunes a Whisper model for speech-to-text transcription in English, and evaluates its performance. Let's break down the steps:

1. **Dependencies Installation**: Installing necessary packages like `accelerate` and `transformers`, along with previously installed dependencies.
2. **Mounting Drive**: Mounting Google Drive to access data.
3. **Library Imports**: Importing required libraries such as pandas, numpy, and transformers.
4. **Whisper Model Components**: Importing and setting up components of the Whisper model like `WhisperFeatureExtractor`, `WhisperTokenizer`, and `WhisperProcessor`.
5. **Loading Dataset**: Loading preprocessed datasets for training, validation, and testing.
6. **Data Collation**: Defining a custom data collator for padding sequences.
7. **Loading Word Error Rate (WER) Metric**: Loading the Word Error Rate metric for evaluation.
8. **Loading Pretrained Whisper Model**: Loading a pre-trained Whisper model for conditional generation.
9. **Configuring Model for English Language Generation**: Configuring the Whisper model for English language generation.
10. **Defining Training Arguments**: Defining training arguments using `Seq2SeqTrainingArguments`, specifying training parameters and settings.
11. **Initializing Trainer**: Initializing the Seq2SeqTrainer for training the model.
12. **Training the Model**: Starting the model training using `trainer.train()`.
13. **Saving Trained Model**: Saving the trained model to a defined path in Google Drive.
14. **Evaluation on Test Dataset**: Evaluating the trained model on the test dataset using `trainer.evaluate()`.

Overall, this script performs end-to-end training, fine-tuning, and evaluation of a Whisper model for speech-to-text transcription in English.

### 4. Inference

This code demonstrates how to perform inference using a pre-trained Whisper model for speech-to-text transcription. Here are the steps outlined in the code:

1. **Install Dependencies**: Installing required dependencies, in this case, the `datasets` library.
2. **Mount Google Drive**: Mounting Google Drive to access relevant files and models.
3. **Define Dataset Path**: Defining the path to the created dataset.
4. **Load Trained Model and Components**: Loading the trained Whisper model, processor, and feature extractor from the specified folder.
5. **Load Audio File**: Loading an audio file for transcription. The file path is specified here.
6. **Extract Audio Features**: Using the feature extractor to extract features from the audio file.
7. **Preprocess Audio Data**: Preprocessing the extracted audio features, including padding if necessary.
8. **Initialize Model and Perform Inference**: Initializing the model and performing inference to transcribe the audio.
9. **Decode Transcribed Text**: Decoding the transcribed text from the model's output.
10. **Display Transcribed Text**: Printing the transcribed text obtained from the model's output.

Additionally, there's an alternative approach shown where inference is performed using a pre-trained Whisper model without fine-tuning. This demonstrates how to directly use a pre-trained model for inference without any additional training.

## Installation

- Ensure all dependencies are installed as listed in each notebook's 'Adding Dependencies' section.
- Data must be accessible either locally or via Google Drive as specified in the notebooks.

- Cleaned data in pickle format, you can skip the notebook 1 if you start with this: [Link](https://drive.google.com/file/d/1UT2gm0I0XY_7ykfBPFKLljAhRHO59n-z/view?usp=drive_link)


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
