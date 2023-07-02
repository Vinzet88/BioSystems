# OpenCV Face Recognition Proof of Concept
### Installation
In order to use these scripts you need to install, via pip or whatever python package manager you prefer, the following libraries:
- numpy
- pillow
- cmake
- opencv-python
- python3-cv2
and be sure to have haarcascade_frontalface_default.xml in your project folder.

### Usage
In order to use the software, please follow this execution order:
- 01_dataset: to create first dataset through webcam acquisition
- 02_training: to train the recognizer 
- 03_recognition: this script start a webcam session during which visitors that we have stored during first step will be recognized (hopefylly).
- 04_test_dataset: as during first step, in this phase we create a new dataset to perform FAR and FRR computation.
- 05_testing: FAR and FRR computation with 2 datasets (test_dataset & test_dataset2 folders)