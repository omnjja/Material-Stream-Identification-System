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

#count images
def count_images():
    images_counts = {}
    for this_class in DATA_CLASSES:
        folder_path = os.path.join(DATA_PATH, this_class)
        images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
        images_counts[this_class] = len(images)
    return images_counts


# if name == "main":
#     print("Image count per class:")
#     print(count_images())


#show random images
def show_data_samples(class_name, n=4):
    folder_path = os.path.join(DATA_PATH, class_name)

    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith(".jpg")
    ]

    chosen_sample = random.sample(images, n)

    plot.figure(figsize=(8, 8))
    for i, image_name in enumerate(chosen_sample):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue  # skip any file has a problem

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plot.subplot(2, 2, i + 1)
        plot.imshow(image)
        plot.title(class_name)
        plot.axis("off")

    plot.show()

show_data_samples("Glass")