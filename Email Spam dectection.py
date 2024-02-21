import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Example data: True and predicted labels for spam classification
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])  # Actual classes (1 for spam, 0 for not spam)
y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])  # Predicted classes

# Creating a confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Visualizing the confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Not Spam', 'Predicted Spam'],
            yticklabels=['Actual Not Spam', 'Actual Spam'])

# Adding labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Spam Classification')

# Displaying the plot
plt.show()
