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
    for phase in ["train", "valid", "test"]:
        for subdir in ["images", "labels"]:
            os.makedirs(os.path.join(dest_dir, phase, subdir), exist_ok=True)

    # Ottieni l'elenco delle sottodirectory (colture)
    crops = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    # Dividi i dati in train, validation e test
    for crop in crops:
        crop_dir = os.path.join(src_dir, crop)
        for root, dirs, files in os.walk(crop_dir):
            # Filtra solo i file immagine
            images = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            random.shuffle(images)

            # Calcola gli indici per la divisione
            train_size = int(len(images) * split_ratio[0])
            valid_size = int(len(images) * split_ratio[1])
            test_size = len(images) - train_size - valid_size

            # Processa ogni immagine
            for i, image in enumerate(images):
                src_img = os.path.join(root, image)

                if i < train_size:
                    dest_phase = "train"
                elif i < train_size + valid_size:
                    dest_phase = "valid"
                else:
                    dest_phase = "test"

                dest_img = os.path.join(dest_dir, dest_phase, "images", image)
                dest_label = os.path.join(dest_dir, dest_phase, "labels", os.path.splitext(image)[0]  + ".txt")

                # Verifica che l'immagine esista prima di copiarla
                if os.path.exists(src_img):
                    shutil.copy(src_img, dest_img)

                    # Genera un'etichetta YOLO per tutte le immagini (coltura = classe 0)
                    with open(dest_label, "w") as f:
                        f.write("0 0.5 0.5 1.0 1.0\n")  # Etichetta YOLO standard
                else:
                    print(f"Immagine non trovata: {src_img}")


# Esegui lo script
src_dir = "./datasets/colture/"  # Directory sorgente
dest_dir = "./datasets/yolo_colture"  # Directory di destinazione
reorganize_dataset(src_dir, dest_dir)
