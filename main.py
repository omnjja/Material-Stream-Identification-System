import os
from test import predict  # import your predict function

# def print_accuracy(y_true, y_pred):
#     correct = 0
#     for i in range(len(y_true)):
#         if y_true[i] == y_pred[i]:
#             correct += 1
#     accuracy = correct / len(y_true)
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#
# y_true = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash", "Unknown"]
data = "test_data"

for filename in os.listdir(data):
    print(filename)

print("start test")
preds = predict(data, "svm_model.pkl")
labels = [CLASSES[p] for p in preds]
print("Predicted labels:", labels)
# print_accuracy(y_true, labels)
print("end test")
