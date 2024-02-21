from sklearn.metrics import precision_recall_curve

# Example data
precision, recall, _ = precision_recall_curve(y_true, y_pred)

# Plotting the precision-recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Displaying the plot
plt.show()
