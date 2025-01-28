import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Percorsi
csv_file = "./datasets/colture/Crop_details.csv"  # Percorso del CSV
base_dir = "./datasets/colture"              # Directory base corretta
output_dir = "./datasets/yolo_colture"    # Directory di output

# Cartelle di output
train_images_dir = os.path.join(output_dir, "train", "images")
train_labels_dir = os.path.join(output_dir, "train", "labels")
val_images_dir = os.path.join(output_dir, "val", "images")
val_labels_dir = os.path.join(output_dir, "val", "labels")

# Creazione delle cartelle
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Leggi il CSV
data = pd.read_csv(csv_file)

# Correggi i percorsi
def fix_path(path):
    # Individua la parte del percorso che inizia da "kag2/" e correggilo
    relative_path = path.split("kag2/", 1)[-1]  # Prende tutto dopo "kag2/"
    return os.path.join(base_dir, relative_path)

data["corrected_path"] = data["path"].apply(fix_path)

# Divisione in train e validation (80-20)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Funzione per processare immagini e creare annotazioni
def process_data(data, images_dir, labels_dir, class_label=0):
    for _, row in data.iterrows():
        image_path = row["corrected_path"]
        
        # Copia l'immagine nella cartella corretta
        if os.path.exists(image_path):  # Verifica che il file esista
            dest_image_path = os.path.join(images_dir, os.path.basename(image_path))
            shutil.copy(image_path, dest_image_path)

            # Crea il file di label (con etichetta fissa "0")
            label_file = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file)
            
            # Scrive solo la classe, senza bounding box
            with open(label_path, "w") as f:
                f.write(f"{class_label}\n")
        else:
            print(f"Attenzione: Immagine non trovata {image_path}")

# Processa i dati di training e validation
process_data(train_data, train_images_dir, train_labels_dir, class_label=0)
process_data(val_data, val_images_dir, val_labels_dir, class_label=0)

print(f"Dataset YOLO creato in: {output_dir}")
