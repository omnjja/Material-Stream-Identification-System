import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps

dataset = "dataset"
augmented_dataset = "augmented_dataset"
target_count = 1000

# get all classes in the dataset
classes = [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset,d))]
# make augmented class for each class in the dataset
for this_class in classes:
    os.makedirs(os.path.join(augmented_dataset, this_class), exist_ok=True)



def random_rotate(img, angle_range=(-30, 30)):
    angle = random.randint(*angle_range)
    return img.rotate(angle)

def flip_lr(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def flip_tb(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def random_brightness(img, factor_range=(0.5, 1.7)):
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)

def random_contrast(img, factor_range=(0.5, 1.7)):
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)

def random_scale(img, scale_range=(0.7, 1.5)):
    scale_factor = random.uniform(*scale_range)
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    return img.resize((new_width, new_height))

def random_crop(img, max_percent=15):
    try:
        max_crop_w = int(img.width * max_percent / 100)
        max_crop_h = int(img.height * max_percent / 100)

        left = random.randint(0, max_crop_w)
        top = random.randint(0, max_crop_h)
        right = img.width - random.randint(0, max_crop_w)
        bottom = img.height - random.randint(0, max_crop_h)

        cropped_image = img.crop((left, top, right, bottom))

        cropped_image = cropped_image.resize(img.size)
        return cropped_image
    except:
        return img


def augment_image(img):
    augmented_images = []
    augmented_images.append(random_rotate(img)) # works
    augmented_images.append(flip_lr(img)) # works
    augmented_images.append(flip_tb(img)) # works
    augmented_images.append(random_brightness(img))
    augmented_images.append(random_contrast(img))
    augmented_images.append(random_scale(img))
    augmented_images.append(random_crop(img))
    return augmented_images

for this_class in classes:
    class_path = os.path.join(dataset, this_class)
    augmented_class_path = os.path.join(augmented_dataset, this_class)
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_name in images:
        src_path = os.path.join(class_path, img_name)
        dst_path = os.path.join(augmented_class_path, img_name)
        try:
            shutil.copy(src_path, dst_path)
        except Exception as e:
            print(f"Warning: Skipping original file {src_path} -> {e}")

    current_count = len(images)
    i = 0
    while current_count < target_count:
        img_name = images[i % len(images)]
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Skipping file {img_path} -> {e}")
            i += 1
            continue

        augmented_imgs = augment_image(img)

        for aug_img in augmented_imgs:
            if current_count >= target_count:
                break
            save_name = f"{os.path.splitext(img_name)[0]}_aug_{current_count}.jpg"
            try:
                aug_img.save(os.path.join(augmented_class_path, save_name))
                current_count += 1
            except Exception as e:
                print(f"Warning: Could not save augmented image -> {e}")
        i += 1

    print(f"Class '{this_class}' done! Total images: {current_count}")

print("Data Augmentation Completed!")