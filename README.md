# Image Feature Matching Web Application

A web application that demonstrates different feature matching algorithms for comparing two images. The app uses three different approaches to detect and match features between images:

- **ORB (Oriented FAST and Rotated BRIEF)**
- **SIFT (Scale-Invariant Feature Transform) with BFMatcher**
- **SIFT with FLANN (Fast Library for Approximate Nearest Neighbors)**

## Features
- Upload two images for comparison
- View matching features using three different algorithms
- Compare the number of matches found by each method
- Pre-loaded example images for quick testing
- Interactive web interface built with Gradio

## Installation

### Clone the Repository
```bash
git clone https://huggingface.co/spaces/JohnJoelMota/Image-Feature-Matching
cd Image-Feature-Matching
```
Install the Required Dependencies
```bash
pip install -r requirements.txt
```
## Usage

### Run the Application Locally
1. Navigate to the project directory where you cloned the repository.
   ```bash
   cd Image-Feature-Matching
   
### Start the application:
python app.py

Open your web browser and navigate to the local URL displayed in the terminal (typically http://localhost:7860).


## Access the Hugging Face Application Online

You can also access the application online using the following link:

https://huggingface.co/spaces/JohnJoelMota/Image-Feature-Matching

In the online version, you can upload your own image pairs or use the provided example images to test the feature matching algorithms.

## Technical Details

## Algorithms Used

### ORB (Oriented FAST and Rotated BRIEF)
- **Description**: Fast and efficient feature detection.
- **Matching**: Uses Hamming distance for matching.
- **Output**: Displays the top 10% of matches.

### SIFT with BFMatcher
- **Description**: Scale and rotation invariant feature detection.
- **Matching**: Uses Brute Force matcher with ratio test.
- **Threshold Ratio**: 0.75.

### SIFT with FLANN
- **Description**: Optimized for faster matching with large datasets.
- **Matching**: Uses KD-tree algorithm.
- **Threshold Ratio**: 0.7.

## Dependencies
- `opencv-python-headless`
- `numpy`
- `matplotlib`
- `gradio`

## Configuration

### Hugging Face Spaces Configuration
- **SDK**: Gradio
- **SDK Version**: 5.4.0
- **Theme Colors**: Purple to Gray
- **License**: Apache 2.0

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
