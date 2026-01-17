# Generated from: Clean_Version.ipynb
# Converted at: 2026-01-17T13:44:36.878Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import zipfile

zip_path = "clean_data.zip"      # replace with your real ZIP file name
extract_to = "PI"               # "." means extract into current folder

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extraction complete!")


import tensorflow as tf
import matplotlib.pyplot as plt
import os

print("TensorFlow version:", tf.__version__)


import sys
print(sys.version)


data_dir = 'PI/data' 


import os
import imghdr

def is_image_file(filepath):
    img_type = imghdr.what(filepath)
    return img_type in ["jpeg", "jpg", "png", "bmp", "gif"]

valid_files = []
for root, _, files in os.walk(data_dir):
    for file in files:
        path = os.path.join(root, file)
        if is_image_file(path):
            valid_files.append(path)
        else:
            print("❌ Not an image:", path)


image_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    validation_split=None,
    interpolation='nearest'
)


import os
import tensorflow as tf

data_dir = "PI/data"
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

def is_valid_image(path):
    try:
        img = tf.io.read_file(path)
        _ = tf.image.decode_image(img, expand_animations=False)
        return True
    except:
        return False

removed = 0

for root, dirs, files in os.walk(data_dir):
    for file in files:
        path = os.path.join(root, file)

        # skip non-image extensions immediately
        if not file.lower().endswith(valid_extensions):
            print("Removed (bad extension):", path)
            os.remove(path)
            removed += 1
            continue

        # try loading the image
        if not is_valid_image(path):
            print("Removed (corrupt image):", path)
            os.remove(path)
            removed += 1

print("Cleaning done. Removed", removed, "invalid files.")


import cv2
import imghdr

image_exts = ['jpeg','jpg', 'bmp', 'png']


import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory('PI/data')


data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

classes = [
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
]
print(classes)

batch[0].shape

import os
import tensorflow as tf

data_dir = "PII/data"
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

def is_valid_image(path):
    try:
        img = tf.io.read_file(path)
        _ = tf.image.decode_image(img, expand_animations=False)
        return True
    except:
        return False

removed = 0

for root, dirs, files in os.walk(data_dir):
    for file in files:
        path = os.path.join(root, file)

        # skip non-image extensions immediately
        if not file.lower().endswith(valid_extensions):
            print("Removed (bad extension):", path)
            os.remove(path)
            removed += 1
            continue

        # try loading the image
        if not is_valid_image(path):
            print("Removed (corrupt image):", path)
            os.remove(path)
            removed += 1

print("Cleaning done. Removed", removed, "invalid files.")


data_dir = "PI/data"

img_size = (256, 256)
batch_size = 32

train = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)


g, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

AUTOTUNE = tf.data.AUTOTUNE
train = train.prefetch(buffer_size=AUTOTUNE)
val = val.prefetch(buffer_size=AUTOTUNE)


preprocess_input = tf.keras.applications.resnet.preprocess_input

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
])


num_classes = 22

inputs = Input(shape=(256, 256, 3))

x = data_augmentation(inputs)   # 👈 augmentation happens here
x = preprocess_input(x)         # 👈 then preprocessing


# -------------------------
# PRETRAINED BASE
# -------------------------
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=x
)
base_model.trainable = False   # freeze weights

# -------------------------
# CUSTOM CLASSIFIER
# -------------------------
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


import os
import tensorflow as tf

data_dir = "PI/data"

def check_image(path):
    try:
        raw = tf.io.read_file(path)
        _ = tf.image.decode_image(raw, expand_animations=False)
        return True
    except:
        return False

bad_files = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        full_path = os.path.join(root, file)

        # skip unsupported types early
        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            bad_files.append(full_path)
            continue

        if not os.path.exists(full_path):
            bad_files.append(full_path)
            continue

        if not check_image(full_path):
            bad_files.append(full_path)

print("Bad files found:", len(bad_files))
for f in bad_files:
    print("❌", f)


class_descriptions = {

    "Akhenaten": (
        "Akhenaten was an ancient Egyptian pharaoh of the 18th Dynasty, known for "
        "introducing religious reforms focused on the worship of the Aten and for "
        "his distinctive artistic style."
    ),

    "amenhotep iii and tiye": (
        "Amenhotep III and Queen Tiye were a powerful royal couple of ancient Egypt. "
        "Their reign was marked by prosperity, artistic achievement, and strong diplomacy."
    ),

    "Bent pyramid for senefru": (
        "The Bent Pyramid of Pharaoh Sneferu is an early pyramid at Dahshur, famous for "
        "its unusual bent shape caused by a change in construction angle."
    ),

    "bust of ramesses ii": (
        "This bust represents Pharaoh Ramesses II, one of Egypt’s greatest rulers, "
        "known for his long reign, military campaigns, and monumental building projects."
    ),

    "Colossal Statue of Ramesses II": (
        "The Colossal Statue of Ramesses II depicts the pharaoh on a massive scale, "
        "symbolizing royal power, divine authority, and the grandeur of New Kingdom Egypt."
    ),

    "Colossoi of Memnon": (
        "The Colossoi of Memnon are two gigantic stone statues of Pharaoh Amenhotep III, "
        "standing near Luxor and once guarding the entrance to his mortuary temple."
    ),

    "Goddess Isis with her child": (
        "This depiction shows the goddess Isis nurturing her child Horus, symbolizing "
        "motherhood, protection, and divine kingship in ancient Egyptian religion."
    ),

    "Hatshepsut": (
        "Hatshepsut was one of the few female pharaohs of ancient Egypt, known for "
        "her successful reign, ambitious building projects, and peaceful trade expeditions."
    ),

    "head Statue of Amenhotep iii": (
        "This statue head portrays Pharaoh Amenhotep III, reflecting the refined artistic "
        "style and prosperity of Egypt during his reign in the 18th Dynasty."
    ),

    "Khafre Pyramid": (
        "The Pyramid of Khafre is one of the three great pyramids at Giza, built for Pharaoh "
        "Khafre and notable for appearing taller due to its elevated foundation."
    ),

    "King Thutmose III": (
        "Thutmose III was a powerful warrior pharaoh of the 18th Dynasty, often called "
        "the 'Napoleon of ancient Egypt' for his military conquests."
    ),

    "Mask of Tutankhamun": (
        "The golden funerary mask of Tutankhamun is one of the most famous artifacts "
        "of ancient Egypt, symbolizing royal wealth, craftsmanship, and the afterlife."
    ),

    "menkaure pyramid": (
        "The Pyramid of Menkaure is the smallest of the three main pyramids at Giza, "
        "built for Pharaoh Menkaure and distinguished by its refined construction."
    ),

    "Nefertiti": (
        "Nefertiti was a queen of ancient Egypt and the wife of Akhenaten, renowned "
        "for her beauty and iconic bust that exemplifies Amarna art."
    ),

    "Pyramid_of_Djoser": (
        "The Pyramid of Djoser, also known as the Step Pyramid, is the earliest large-scale "
        "stone structure in Egypt and a major milestone in pyramid construction."
    ),

    "Ramessum": (
        "The Ramesseum is the mortuary temple of Pharaoh Ramesses II, built to honor "
        "his reign and achievements in ancient Thebes."
    ),

    "sphinx": (
        "The Great Sphinx of Giza is a monumental limestone statue with the body of a lion "
        "and the head of a pharaoh, symbolizing strength and royal power."
    ),

    "Statue of King Zoser": (
        "This statue represents King Djoser, the ruler associated with the Step Pyramid "
        "and one of the earliest monumental figures in Egyptian sculpture."
    ),

    "Statue of Tutankhamun with Ankhesenamun": (
        "This statue depicts Pharaoh Tutankhamun alongside his wife Ankhesenamun, "
        "symbolizing royal partnership and divine legitimacy."
    ),

    "Temple_of_Isis_in_Philae": (
        "The Temple of Isis at Philae was a major religious center dedicated to the goddess "
        "Isis and is renowned for its well-preserved reliefs and architecture."
    ),

    "Temple_of_Kom_Ombo": (
        "The Temple of Kom Ombo is a unique double temple dedicated to two sets of gods, "
        "including Sobek and Horus, reflecting dual religious worship."
    ),

    "The Great Temple of Ramesses II": (
        "The Great Temple of Ramesses II at Abu Simbel is a monumental rock temple "
        "celebrating the pharaoh’s power and divine status."
    )

}


def predict_loaded_image(model, img):
    # --- Convert image to array ---
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # --- Predict ---
    preds = model.predict(img_array, verbose=0)[0]
    idx = np.argmax(preds)

    # ✅ DEFINE class_name FIRST
    class_name = class_names[idx]
    confidence = preds[idx] * 100

    # ✅ THEN get description
    description = class_descriptions.get(
        class_name,
        "No description available for this class."
    )

    # --- Display ---
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"{class_name} ({confidence:.2f}%)")
    plt.axis("off")

    plt.figtext(
        0.5, -0.2,
        description,
        wrap=True,
        horizontalalignment="center",
        fontsize=11
    )

    plt.show()

    # --- Console output ---
    print("Predicted class:", class_name)
    print(f"Confidence: {confidence:.2f}%")
    print("Description:", description)


history1 = model.fit(
    train,
    validation_data=val,
    epochs=4
)


IMG_SIZE = (256, 256)   # must match your training size


img = tf.keras.utils.load_img("A.12.jpg", target_size=IMG_SIZE)

import matplotlib.pyplot as plt
plt.imshow(img)
plt.axis("off")
plt.show()


predict_loaded_image(model, img)