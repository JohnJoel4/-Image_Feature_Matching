# Image Feature Matching Web Application

Hugging Face application that demonstrates different feature matching algorithms for comparing two images. The app uses three different approaches to detect and match features between images:

**1. ORB (Oriented FAST and Rotated BRIEF)**

**2. SIFT (Scale-Invariant Feature Transform) with BFMatcher**

**3. SIFT with FLANN (Fast Library for Approximate Nearest Neighbors)**

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
```bash
python app.py
```
Open your web browser and navigate to the local URL displayed in the terminal (typically http://localhost:7860).


## Access the Hugging Face Application Online

You can also access the application online using the following link:

https://huggingface.co/spaces/JohnJoelMota/Image-Feature-Matching

In the online version, you can upload your own image pairs or use the provided example images to test the feature matching algorithms.

## Visualizing Image Matching ResultsInput Images

The application lets you upload or select example images to be processed for feature matching. For instance:

**Input Image 1:**
This is the first input image where features will be detected.

![charminar_02 (1)](https://github.com/user-attachments/assets/672eaece-661c-48f2-ac29-c503be8b43f3)

**Input Image 2:**
This is the second input image for comparison with Image 1.

![charminar_01 (1)](https://github.com/user-attachments/assets/dd2f4fe7-5a32-421c-a788-f361602e03bc)

### Results
After processing, the application displays the results of feature matching using three algorithms:

**ORB Algorithm**
This algorithm highlights detected features and matches between Image 1 and Image 2 using ORB (Oriented FAST and Rotated BRIEF), emphasizing speed and efficiency.

![image](https://github.com/user-attachments/assets/51ea5f28-1d70-4913-b4ef-4d3aae1974ad)

**SIFT Algorithm**
Using SIFT (Scale-Invariant Feature Transform) with a Brute Force matcher, this method shows scale- and rotation-invariant matches between the input images, suitable for detecting complex feature points.

![image](https://github.com/user-attachments/assets/e4db0771-f75d-4391-96c3-8ea7ca9965df)

**SIFT with FLANN Algorithm**
The SIFT with FLANN (Fast Library for Approximate Nearest Neighbors) option is optimized for faster matching, ideal for large datasets. This result shows the matched points using the KD-tree algorithm, allowing quick and approximate matching.

![image](https://github.com/user-attachments/assets/16761231-3c08-472e-bf85-80b523241d21)

Each method provides a visual comparison, helping you analyze which algorithm best matches features across the two images.

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
