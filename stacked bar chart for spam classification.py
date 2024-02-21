import numpy as np
import matplotlib.pyplot as plt

# Example data
true_positive, false_negative, false_positive, true_negative = conf_matrix.ravel()

# Plotting a stacked bar chart
plt.bar(['Actual Positive', 'Actual Negative'], [true_positive + false_negative, false_positive + true_negative], color=['blue', 'green'], label='Predicted Positive')
plt.bar(['Actual Positive', 'Actual Negative'], [false_negative, true_negative], color=['lightblue', 'lightgreen'], label='Predicted Negative', bottom=[true_positive, false_positive])

# Adding labels and title
plt.xlabel('Actual')
plt.ylabel('Count')
plt.title('Stacked Bar Chart for Spam Classification')
plt.legend()

# Displaying the plot
plt.show()
