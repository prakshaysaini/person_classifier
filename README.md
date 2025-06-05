A machine learning model to classify images of people, trained using a Support Vector Machine (SVM).

🔧 Steps:
Data Cleaning

    Detect faces and crop them.

    Detect a valid pair of eyes to ensure the face is usable.

    Manually clean the cropped images to remove incorrect data (e.g., images of other people).

Feature Extraction

    Apply wavelet transformation to the cropped images to help the model understand facial features more effectively.

Model Creation

    Build a classification model using SVM.

Model Training

    Train the model on the processed and feature-extracted images.
