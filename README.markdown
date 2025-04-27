# Audio Command Classification with TensorFlow

## Description
This project demonstrates how to classify audio commands using a convolutional neural network (CNN) with TensorFlow. It uses the "mini Speech Commands" dataset, a subset of the Speech Commands dataset, which contains audio clips of eight different words: "no", "yes", "down", "go", "left", "up", "right", and "stop". The audio data is transformed into spectrograms, which are then used as input for the CNN to perform classification.

## Installation
To run this project, you need to have the following dependencies installed:

- TensorFlow
- TensorFlow Datasets
- Matplotlib
- Seaborn

You can install these dependencies using pip:

```bash
pip install tensorflow tensorflow-datasets matplotlib seaborn
```

## Usage
1. **Download the Dataset**:
   - The "mini Speech Commands" dataset can be loaded directly in the notebook using `tfds.load('speech_commands', split='train', as_supervised=True)`. Ensure you have TensorFlow Datasets installed.

2. **Run the Notebook**:
   - Open the Jupyter Notebook (`simple_audio(1).ipynb`) in your preferred environment (e.g., Google Colab, JupyterLab).
   - Ensure all dependencies are installed.
   - Execute the cells in order to load the data, preprocess it, train the model, and evaluate its performance.

## Dataset
The project utilizes the "mini Speech Commands" dataset, a subset of the Speech Commands dataset. It contains over 105,000 audio files, each approximately 1 second long, recorded at a 16kHz sampling rate. The dataset is split into training, validation, and test sets. The dataset can be accessed via [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/speech_commands).

## Classification Process
The classification process involves the following steps:

1. **Data Preparation**:
   - Audio files are loaded and standardized to 1 second in length at a 16kHz sampling rate. Shorter clips are padded to 16,000 samples.
   - The dataset is split into training, validation, and test sets.

2. **Feature Extraction**:
   - Raw audio waveforms are converted into spectrograms using the Short-Time Fourier Transform (STFT). The STFT parameters are set to produce nearly square spectrograms (e.g., shape `(124, 129, 1)`), where:
     - 124 represents the number of time frames.
     - 129 represents the number of frequency bins.
     - 1 is the channels dimension (added for CNN compatibility).
   - Spectrograms are resized to 32x32 and normalized to ensure consistency.

3. **Model Training**:
   - A CNN model is used for classification. The model architecture includes:
     - Preprocessing layers (resizing and normalization).
     - Convolutional layers with max pooling.
     - Dropout layers for regularization.
     - Dense layers for classification.
   - The model is trained using the Adam optimizer and sparse categorical crossentropy loss.

4. **Evaluation**:
   - The model's performance is evaluated on the test set using metrics like accuracy and a confusion matrix.

## Features Used for Classification
The primary features used for classification are **spectrograms**, which are 2D representations of the audio's frequency content over time. Specifically:
- **Spectrograms**:
  - Generated from audio waveforms using STFT.
  - Capture frequency components and their changes over time, which are crucial for distinguishing between spoken commands.
- **Preprocessing**:
  - Spectrograms are resized to 32x32 and normalized to standardize input for the CNN.

## Visualizations
The notebook includes visualizations of waveforms and spectrograms to aid in understanding the data and the feature extraction process. These visualizations help in exploring the dataset and verifying the quality of the spectrograms.

## Model Export
The notebook includes code to export the trained model with preprocessing steps included. This allows users to directly input audio files or waveforms into the exported model without needing to preprocess them separately, making it easier to deploy the model for inference.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or feedback, please contact [Your Name/Email].