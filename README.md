# Rice Disease Detection System

## Overview

A deep learning-based web application for detecting rice leaf diseases from images. Users can upload a rice leaf image through the web interface, and the system predicts the disease category along with a confidence score.

## Diseases Detected

* Bacterial Blight
* Blast
* Brown Spot

## Features

* Rice leaf image upload
* Deep learning-based disease classification
* Confidence score prediction
* Disease information display
* Web-based user interface
* Flask API backend

## Tech Stack

* Python
* Flask
* TensorFlow / Keras
* EfficientNetV2
* NumPy
* Pillow (PIL)
* HTML
* CSS
* JavaScript

## Project Structure

```text
Rice-Disease-Detection/
│
├── models/
├── static/
├── templates/
├── app.py
├── README.md
└── .gitignore
```

## How It Works

1. Upload a rice leaf image.
2. The image is preprocessed and resized.
3. The trained EfficientNetV2 model performs classification.
4. The predicted disease and confidence score are displayed.
5. Additional disease information is shown to the user.

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

## Future Improvements

* Support more rice diseases
* Mobile-friendly interface
* Disease severity estimation
* Cloud deployment

## Disclaimer

This project is intended for educational and research purposes. Predictions should be verified by agricultural experts before making farming decisions.
