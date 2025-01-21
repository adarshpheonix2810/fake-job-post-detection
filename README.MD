# Fake Job Post Detection

## Overview
This project focuses on detecting fake job posts using machine learning. Fake job advertisements are often created to scam individuals by stealing personal information or money. This project trains a machine learning model to classify job posts as either fake or genuine based on their textual content.

---

## Features
- Predict whether a job post is **fake** or **genuine**.
- Provides a **Graphical User Interface (GUI)** for user-friendly interaction.
- Utilizes a **TF-IDF vectorizer** for feature extraction and a **Random Forest Classifier** for prediction.
- Includes model training, testing, and evaluation.

---

## Dataset
The project uses the **Fake Job Postings** dataset. Ensure the dataset file (`fake_job_postings.csv`) is in the working directory.

---

## Requirements
### Dependencies
The project requires the following Python libraries:
- pandas
- numpy
- scikit-learn
- joblib
- tkinter

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn
```

### Files Required
1. `fake_job_postings.csv`: Dataset file.
2. `fake_job_post_detector.pkl`: Trained machine learning model.
3. `tfidf_vectorizer.pkl`: Trained TF-IDF vectorizer.

---

## Project Structure
```
fake-job-post-detection/
|
├── fake_job_postings.csv        # Dataset file
├── model_training.py            # Script for training the model
├── fake_job_gui.py              # GUI application script
├── fake_job_post_detector.pkl   # Trained model
├── tfidf_vectorizer.pkl         # TF-IDF vectorizer
├── README.md                    # Documentation
```

---

## Steps to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/adarshpheonix2810/fake-job-post-detection.git
cd fake-job-post-detection
```

### 2. Prepare the Dataset
Ensure the dataset file (`fake_job_postings.csv`) is in the root directory of the project.

### 3. Train the Model
Run the `model_training.py` script to train the model and save the required files:
```bash
python model_training.py
```
This will generate:
- `fake_job_post_detector.pkl`
- `tfidf_vectorizer.pkl`

### 4. Run the GUI
Start the GUI application using the following command:
```bash
python fake_job_gui.py
```
This will launch a window where you can input job descriptions and get predictions.

---

## Model Training Pipeline
1. **Preprocessing**:
   - Handles missing data.
   - Extracts the `description` column as the input and `fraudulent` column as the target.

2. **Feature Engineering**:
   - Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features.

3. **Model**:
   - A Random Forest Classifier is used for prediction.

4. **Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-Score.

---

## GUI Features
- **Input Field**: Users can enter a job description.
- **Predict Button**: Triggers the prediction.
- **Output**: Displays whether the job post is "Fake" or "Genuine" along with the confidence level.

---

## Future Improvements
- Implement advanced NLP models like BERT for better predictions.
- Add support for detecting fake posts in multiple languages.
- Integrate the project into a web application for broader accessibility.

---

## Contributing
Contributions are welcome! If you'd like to make improvements or fix issues, feel free to create a pull request or open an issue in the repository.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

