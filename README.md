# Real-time Face Recognition using Haar-Cascades and LBPH-Face-Recognizer

## To run the code, please either use "Visual Studio" or "Jupyter Notebook from Anaconda Navigator".

### Thank you.

<br>

## Code Explanation:

1. **Importing Libraries**: The code starts by importing the necessary libraries, including `cv2` for OpenCV operations, `os` for file operations, and `numpy` for numerical computations.

2. **Defining Paths and Variables**: The variable `training_data_path` stores the path to the directory containing training images. It initializes LBPH (Local Binary Patterns Histograms) face recognizer and loads the pre-trained frontal face classifier using the Haar cascade classifier.

3. **Preparing Training Data**: The function `prepare_training_data()` reads images from the training data directory, detects faces using the Haar cascade classifier, and prepares the training data by extracting faces, assigning labels, and creating a label map.

4. **Training the Recognizer**: The LBPH face recognizer is trained using the extracted faces and labels using the `train()` method.

5. **Capturing Video**: The code initializes a video capture object using `cv2.VideoCapture(0)` to capture frames from the default camera (index 0).

6. **Face Recognition Loop**: Inside the main loop, the code continuously captures frames from the video feed, converts each frame to grayscale, and detects faces using the Haar cascade classifier.

7. **Recognizing Faces**: For each detected face, the code extracts the region of interest (ROI), predicts the label using the trained recognizer, and draws a rectangle around the face with the predicted label as text.

8. **Displaying Results**: The processed frame with rectangles and labels is displayed in a window titled "Real-time Face Recognition" using `cv2.imshow()`.

9. **Exiting**: The program exits the loop if the 'q' key is pressed, releasing the video capture and closing all OpenCV windows.

*** **roi** -> region of interest (face).

*** **In pose detection**, there are many roi such as face, nose, neck, shoulder, etc.

## Key Points:

- Utilizes LBPH face recognizer and Haar cascade classifier for face recognition.
- Reads training data from a directory and prepares it for training.
- Trains the recognizer using the prepared training data.
- Captures video from the default camera and performs real-time face recognition.
- Displays the recognized faces with bounding boxes and labels.
- Allows quitting the application by pressing the 'q' key.
