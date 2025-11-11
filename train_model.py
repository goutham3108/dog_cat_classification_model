import os
import shutil
import requests
import tarfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def download_file_robust(url, target_path, min_size_mb=1):
    """
    Downloads a file robustly. Checks for existence and size, and deletes
    incomplete files before re-downloading.
    """
    if os.path.exists(target_path) and os.path.getsize(target_path) > min_size_mb * 1024 * 1024:
        print(f"'{os.path.basename(target_path)}' already exists and is valid. Skipping download.")
        return True

    print(f"Downloading '{os.path.basename(target_path)}'... This may take a few minutes.")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return False

    if os.path.getsize(target_path) < min_size_mb * 1024 * 1024:
        print(f"Error: Downloaded file '{target_path}' is incomplete. Please check your network and try again.")
        os.remove(target_path)
        return False

    return True


def extract_tar(tar_path, extract_path='.'):
    """Extracts a .tar.gz file with error handling."""
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_path, filter='data')
        return True
    except (tarfile.ReadError, EOFError) as e:
        print(f"Error extracting '{tar_path}': {e}. The file may be corrupted.")
        return False

def train_and_save_model():
    """
    This function encapsulates the entire model training pipeline,
    from data download to saving the final model.
    """
    print("--- Starting Dataset Download and Preparation ---")

    content_dir = 'content'
    os.makedirs(content_dir, exist_ok=True)
    
    images_url = 'http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
    annotations_url = 'http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'
    
    images_tar_path = os.path.join(content_dir, 'images.tar.gz')
    annotations_tar_path = os.path.join(content_dir, 'annotations.tar.gz')

    if not download_file_robust(images_url, images_tar_path, min_size_mb=750):
        return
    if not download_file_robust(annotations_url, annotations_tar_path, min_size_mb=10):
        return
    
    print("Extracting datasets...")
    if not extract_tar(images_tar_path, content_dir) or not extract_tar(annotations_tar_path, content_dir):
        return
    print("Dataset downloaded and extracted successfully.")

    base_dir = os.path.join(content_dir, 'dataset')
    images_dir = os.path.join(content_dir, 'images')
    annotations_file = os.path.join(content_dir, 'annotations/list.txt')

    os.makedirs(os.path.join(base_dir, 'cat'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'dog'), exist_ok=True)

    print("Structuring dataset into 'cat' and 'dog' folders...")
    with open(annotations_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split()
            image_name = parts[0] + '.jpg'   # ✅ Fixed line
            species_id = int(parts[2])       # 1: Cat, 2: Dog

            source_path = os.path.join(images_dir, image_name)
            
            if os.path.exists(source_path):
                if species_id == 1:  # Cat
                    destination_path = os.path.join(base_dir, 'cat', image_name)
                elif species_id == 2:  # Dog
                    destination_path = os.path.join(base_dir, 'dog', image_name)
                else:
                    continue
                
                shutil.move(source_path, destination_path)

    print(" Dataset successfully structured.")
    shutil.rmtree(images_dir)
    shutil.rmtree(os.path.join(content_dir, 'annotations'))

    print("\n--- Building and Optimizing Data Pipeline ---")
    
    IMAGE_SIZE = (160, 160)
    BATCH_SIZE = 32
    DATASET_DIR = base_dir

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    print(" Data pipelines configured.")

    print("\n--- Building the Model via Transfer Learning ---")

    IMG_SHAPE = IMAGE_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    print(" Model architecture built.")
    model.summary()

    print("\n--- Starting Initial Model Training (Feature Extraction) ---")

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    initial_epochs = 10
    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)

    print("\n--- Starting Fine-Tuning Phase ---")

    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                  metrics=['accuracy'])

    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs

    model.fit(train_dataset,
              epochs=total_epochs,
              initial_epoch=history.epoch[-1],
              validation_data=validation_dataset)
    
    print("\n--- Training Complete. Saving Model... ---")

    model.save('cat_dog_classifier.keras')
    print(" Model saved as 'cat_dog_classifier.keras'")

if __name__ == '__main__':
    train_and_save_model()