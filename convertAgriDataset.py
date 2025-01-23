import os
import shutil
import random

def reorganize_dataset(src_dir, dest_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Reorganizza un dataset di immagini in cartelle train, valid e test, gestendo anche sottodirectory.

    Args:
        src_dir: Percorso della directory sorgente.
        dest_dir: Percorso della directory di destinazione.
        split_ratio: Tupla contenente le proporzioni per train, valid e test.
    """

    # Crea le directory di destinazione
    for phase in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(dest_dir, phase, subdir), exist_ok=True)

    # Ottieni l'elenco delle colture
    crops = os.listdir(src_dir)
    crops.remove("Crop_details.csv")  # Rimuovi il file CSV se presente

    # Dividi i dati in train, validation e test
    for crop in crops:
        crop_dir = os.path.join(src_dir, crop)
        for root, dirs, files in os.walk(crop_dir):
            images = [f for f in files if f.endswith(('.jpg', '.png'))]  # Filtra solo immagini
            random.shuffle(images)

            # Calcola gli indici per la divisione
            train_size = int(len(images) * split_ratio[0])
            valid_size = int(len(images) * split_ratio[1])
            test_size = len(images) - train_size - valid_size

            for i, image in enumerate(images):
                src_img = os.path.join(root, image)
                if i < train_size:
                    dest_phase = 'train'
                elif i < train_size + valid_size:
                    dest_phase = 'valid'
                else:
                    dest_phase = 'test'

                dest_img = os.path.join(dest_dir, dest_phase, 'images', image)
                dest_label = os.path.join(dest_dir, dest_phase, 'labels', image[:-4] + '.txt')

                shutil.copy(src_img, dest_img)
                with open(dest_label, 'w') as f:
                    # Sostituisci con la tua logica per creare le etichette
                    f.write(f"0 0.5 0.5 1.0 1.0\n")  # Esempio di etichetta YOLO

# Esegui lo script
src_dir = "./datasets/colture/"
dest_dir = "./datasets/coltureYOLO"
reorganize_dataset(src_dir, dest_dir)