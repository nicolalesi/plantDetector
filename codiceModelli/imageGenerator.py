from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Percorso del dataset originale
original_data_dir = r"D:\Magistrale\ArtificialIntelligence\progettoAI\datasets\pianteecolture\structured_data\train\images"

# Percorso per salvare le immagini aumentate
augmented_data_dir = "./datasets/augmented_data_dir"
os.makedirs(augmented_data_dir, exist_ok=True)

# Configura il generatore di immagini
datagen = ImageDataGenerator(
    rotation_range=60,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Aumenta ogni immagine
for img_name in os.listdir(original_data_dir):
    if img_name.endswith('.jpeg') or img_name.endswith('.png'):
        img_path = os.path.join(original_data_dir, img_name)
        img = load_img(img_path)  # Carica l'immagine
        img_array = img_to_array(img)  # Converte in array
        img_array = img_array.reshape((1,) + img_array.shape)  # Ridimensiona per il generatore

        # Genera immagini aumentate e salvale
        i = 0
        for batch in datagen.flow(img_array, batch_size=1,
                                  save_to_dir=augmented_data_dir,
                                  save_prefix='aug',
                                  save_format='jpeg'):
            i += 1
            if i > 5:  # Ferma dopo aver generato 50 immagini
                break

# Verifica il numero di immagini generate
augmented_images = os.listdir(augmented_data_dir)
print(f"Numero di immagini generate: {len(augmented_images)}")
