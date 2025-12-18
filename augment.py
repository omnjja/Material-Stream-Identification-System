import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps

dataset = "dataset"
augmented_dataset = "augmented_dataset"
TARGET_COUNT = 1000

# get classes in the dataset
DATA_CLASSES = []
for folder in os.listdir(dataset):
    if os.path.isdir(os.path.join(dataset,folder)):
        DATA_CLASSES.append(folder)

# generate augmented class for each class in the dataset
for this_class in DATA_CLASSES:
    os.makedirs(os.path.join(augmented_dataset, this_class), exist_ok=True) # don't generate error if class already exists

### augmentation process ###

# rotate image by random angle
def random_rotate(image, angle_range=(-30, 30)):
    rotation_angle = random.randint(*angle_range)
    return image.rotate(rotation_angle)

# flip image left-right
def flip_horizontally(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

# flip image top-bottom
def flip_vertically(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

# change brightness of image randomly (make it darker or lighter)
def random_brightness(image, brightness_factor_range=(0.5, 1.7)):
    image_enhancer = ImageEnhance.Brightness(image)  # create enhancer object
    brightness_factor = random.uniform(*brightness_factor_range) # choose random brightness factor
    return image_enhancer.enhance(brightness_factor)

# change contrast of image randomly
def random_contrast(image, contrast_factor_range=(0.5, 1.7)):
    image_enhancer = ImageEnhance.Contrast(image)  # create enhancer object
    contrast_factor = random.uniform(*contrast_factor_range)  # choose random contrast factor
    return image_enhancer.enhance(contrast_factor)

# scale the image randomly
def random_scaling(image, scaling_range=(0.7, 1.5)):
    scaling_factor = random.uniform(*scaling_range)  # choose random scaling factor
    image_new_width = int(image.width * scaling_factor)
    # calculate new dimensions of the image
    image_new_height = int(image.height * scaling_factor)
    return image.resize((image_new_width, image_new_height))

# crop part of the image randomly
def random_crop(image, max_crop_percent=15):
    try:
        # max allowed cropped pixels
        max_crop_width = int(image.width * max_crop_percent / 100)
        max_crop_height = int(image.height * max_crop_percent / 100)
        # determine random cropped boundaries
        left = random.randint(0, max_crop_width)
        top = random.randint(0, max_crop_height)
        right = image.width - random.randint(0, max_crop_width)
        bottom = image.height - random.randint(0, max_crop_height)

        cropped_image = image.crop((left, top, right, bottom))
        # resize the cropped image to be the same size as the original one
        cropped_image = cropped_image.resize(image.size)
        return cropped_image
    except:
        return image # return the original image in case of any errors

def apply_image_augmentation(image):
    augmented_images = [random_rotate(image), flip_horizontally(image), flip_vertically(image),
                        random_brightness(image), random_contrast(image), random_scaling(image), random_crop(image)]
    return augmented_images

### apply augmentation process ###

# loop over all classes
for this_class in DATA_CLASSES:
    class_path = os.path.join(dataset, this_class) # get class path
    augmented_class_path = os.path.join(augmented_dataset, this_class) # generate new augmented class path
    # get all image files in the class
    images = []
    for file in os.listdir(class_path):
        if  file.lower().endswith((".png", ".jpg", ".jpeg")):
            images.append(file)

    # copy original image to augmented class folder
    for image_name in images:
        source_path = os.path.join(class_path, image_name)
        destination_path = os.path.join(augmented_class_path, image_name)
        try:
            shutil.copy(source_path, destination_path)
        except Exception as e:
            print(f"Warning: Skipping original image file {source_path} -> {e}")

    current_count = len(images) # number of images in the class
    i = 0
    # do augmentation until the target count reached
    while current_count < TARGET_COUNT:
        image_name = images[i % len(images)]
        image_path = os.path.join(class_path, image_name)
        # open image and convert it to RGB
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Skipping file {image_path} -> {e}")
            i += 1
            continue
        final_augmented_images = apply_image_augmentation(img) # apply augmentation function

        for augmented_image in final_augmented_images:
            if current_count >= TARGET_COUNT: # break if target count reached
                break
            image_save_name = f"{os.path.splitext(image_name)[0]}_augmented_{current_count}.jpg" # create unique save name
            try:
                augmented_image.save(os.path.join(augmented_class_path, image_save_name))
                current_count += 1
            except Exception as e:
                print(f"Warning: Could not save augmented image -> {e}")
        i += 1

    print(f"Class '{this_class}' done! Total images: {current_count}")

print("Data Augmentation Completed!")