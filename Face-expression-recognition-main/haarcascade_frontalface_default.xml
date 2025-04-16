import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Defines emotion categories
emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']

# Generates synthetic data (replace with actual model predictions and ground truth)
true_labels = np.random.choice(emotions, 100) 
predicted_labels = np.random.choice(emotions, 100)  

# Computes confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=emotions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Emotion Classification")
plt.show()
