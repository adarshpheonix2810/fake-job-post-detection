import tkinter as tk
from tkinter import messagebox
import joblib

# Load the trained model and vectorizer
model = joblib.load('fake_job_post_detector.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_fake_job():
    """Predict whether the job post is fake or not."""
    input_text = entry.get("1.0", tk.END).strip()  # Get user input from the text widget
    if not input_text:
        messagebox.showwarning("Input Error", "Please enter a job description!")
        return
    
    # Transform the input using the vectorizer
    input_tfidf = vectorizer.transform([input_text])
    
    # Predict using the model
    prediction = model.predict(input_tfidf)[0]
    probability = model.predict_proba(input_tfidf)[0].max()
    
    # Display the result
    if prediction == 1:
        result = f"The job post is likely FAKE with a confidence of {probability:.2f}."
    else:
        result = f"The job post is likely GENUINE with a confidence of {probability:.2f}."
    
    result_label.config(text=result)

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Fake Job Post Detector")
root.geometry("500x400")

# Add widgets
tk.Label(root, text="Enter Job Description:", font=("Arial", 14)).pack(pady=10)
entry = tk.Text(root, height=10, width=50, font=("Arial", 12))
entry.pack(pady=10)

predict_button = tk.Button(root, text="Predict", font=("Arial", 14), command=predict_fake_job)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=400, justify="center")
result_label.pack(pady=20)

# Run the Tkinter main loop
root.mainloop()
