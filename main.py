import os

from test import predict
print("start test")
CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash", "Unknown"]

data = "test_data"
for filename in os.listdir(data):
    print(filename)

preds = predict("test_data", "svm_model.pkl")
labels = [CLASSES[p] for p in preds]

print(labels)



print("end test")