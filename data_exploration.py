import os
import random
import cv2
import matplotlib.pyplot as plot

DATA_PATH = "dataset"

DATA_CLASSES = [
    "Glass",
    "Paper",
    "Cardboard",
    "Plastic",
    "Metal",
    "Trash"
]

# count images
def count_images():
    count = {}
    for this_class in DATA_CLASSES:
        folder_path = os.path.join(DATA_PATH, this_class)
        images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
        count[this_class] = len(images)
    return count

print("Image count per class:")
print(count_images())

# show random images
IMAGES_COUNT = 4
def explore_sample(class_name, n= IMAGES_COUNT):
    folder_path = os.path.join(DATA_PATH, class_name)
    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".jpg"):
            images.append(file)
    data_sample = random.sample(images, n)

    plot.figure(figsize=(8, 8))
    for i in range(len(data_sample)):
        image_name = data_sample[i]
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue # skip if smth went wrong
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plot.subplot(2, 2, i + 1)
        plot.imshow(image)
        plot.title(class_name)
        plot.axis("off")

    plot.show()

explore_sample("Glass")