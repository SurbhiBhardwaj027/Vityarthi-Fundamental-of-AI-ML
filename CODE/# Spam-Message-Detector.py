# Spam-Message-Detector
# Spam Message Detector - GUI VERSION 
# Using Naive Bayes + TF-IDF + Tkinter UI
# Author: Surbhi Bhardwaj
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# DATASET

data ={
   "message":[
       "Congratulations! You won a free lottery ticket.",
       "Hello, how are you?",
       "Claim your free prize now!",
       "Are we meeting tomorrow?",
       "Please call me later.",
       "Win a brand new car! Limited offer.",
       "Let's have lunch today.",
       "Urgent!Your account has been compromised.",
       "Send me your assignment."
    ],  
    "label": ["spam","ham","spam","ham","spam","ham","spamm","ham","spam"]
}
df = pd.DataFrame(data)

# CLEAN TEXT FUNCTION

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]','',text)
    text = re.sub(r'\\s+','',text)
    return text.strip()
df["cleaned"] = df["message"].apply(clean_text)

# TRAINING MODEL

X_train, X_test,y_train, y_test = train_test_split(
    df["cleaned"],df["label"], test_size=0.3, random_state = 42
)    
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec,y_train)

# GUI SETUP

root = tk.Tk()
root.title("Spam Message Detector - AI & ML Project")
root.geometry("500x400")
root.config(bg="#1B1B1B")

title_label = tk.Label(
    root,
    text="Spam Message Detector",
    font=("Arial", 20 , "bold"),
    fg="white",
    bg="#1B1B1B"
)
title_label.pack(pady=20)

msg_label = tk.Label(
    root,
    text="Enter your message:",
    font=("Arial",12),
    fg="white",
    bg="#1B1B1B"
)
msg_label.pack()

msg_box = tk.Text(root,height=5,width=50,font=("Arial",12))
msg_box.pack(pady=10)

# DETECT SPAM FUNCTION

def detect_spam_gui():
    message = msg_box.get("1.0", tk.END).strip()

    if message == "":
        messagebox.showwarning("Error", "Please enter a message.")
        return

    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]

    if result == "spam":
        messagebox.showerror("Result","🚨This message is SPAM!")
    else:
        messagebox.showinfo("Result", "✅ This message is NOT spam.")

# CLEAR FUNCTION

def clear_text():
    msg_box.delete("1.0", tk.END)

# BUTTONS

btn_frame = tk.Frame (root,bg="#1B1B1B")
btn_frame.pack(pady=20)

detect_button = tk.Button(
    btn_frame,
    text="Detect Spam",
    width=15,
    height=2,
    command=detect_spam_gui
)
detect_button.grid(row=0,column=1,padx=10)

clear_button=tk.Button(
    btn_frame,
    text="Clear",
    width=15,
    height=2,
    command=clear_text
)    
clear_button.grid(row=0, column=1, padx=10)

root.mainloop()

    
    