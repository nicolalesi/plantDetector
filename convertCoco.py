import os
import json
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def coco_to_yolo(coco_json_path, output_dir, images_dir, val_size=0.2, test_size=0.1):
    # Crea le directory di output
    images_output_dir = os.path.join(output_dir, 'train', 'images')
    labels_output_dir = os.path.join(output_dir, 'train', 'labels')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # Carica il JSON COCO
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Mappa ID delle categorie ai nomi delle categorie (opzionale)
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Mappa immagini: image_id -> file_name
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Elaborazione delle annotazioni
    annotations = coco_data['annotations']
    image_ids = list(images.keys())
    train_ids, temp_ids = train_test_split(image_ids, test_size=(val_size + test_size))
    val_ids, test_ids = train_test_split(temp_ids, test_size=(test_size / (val_size + test_size)))

    # Funzione per filtrare immagini e annotazioni
    def filter_by_ids(image_ids, images, annotations):
        filtered_images = [img for img in images if img['id'] in image_ids]
        filtered_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]
        return filtered_images, filtered_annotations

    # Creare set di training, validazione e test
    sets = {
        'train': (train_ids, os.path.join(images_output_dir), os.path.join(labels_output_dir)),
        'val': (val_ids, os.path.join(output_dir, 'val', 'images'), os.path.join(output_dir, 'val', 'labels')),
        'test': (test_ids, os.path.join(output_dir, 'test', 'images'), os.path.join(output_dir, 'test', 'labels'))
    }

    for key, (ids, img_out_dir, label_out_dir) in sets.items():
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)
        
        for image_id, image_name in tqdm({img['id']: img['file_name'] for img in coco_data['images'] if img['id'] in ids}.items(), desc=f"Processing {key} images"):
            # Copia l'immagine
            src_image_path = os.path.join(images_dir, image_name)
            dst_image_path = os.path.join(img_out_dir, image_name)
            shutil.copy(src_image_path, dst_image_path)
            
            # Trova le annotazioni per questa immagine
            image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
            
            # Scrivere il file delle label
            label_file_path = os.path.join(label_out_dir, os.path.splitext(image_name)[0] + '.txt')
            with open(label_file_path, 'w') as label_file:
                for ann in image_annotations:
                    category_id = ann['category_id'] - 1  # YOLO usa categorie da 0
                    bbox = ann['bbox']  # [x_min, y_min, width, height]
                    x_min, y_min, width, height = bbox
                    
                    # Calcolo di YOLO bbox format
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2
                    x_center /= coco_data['images'][image_id]['width']
                    y_center /= coco_data['images'][image_id]['height']
                    width /= coco_data['images'][image_id]['width']
                    height /= coco_data['images'][image_id]['height']
                    
                    # Scrivi la riga nel file delle label
                    label_file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

    print("Conversione completata.")

# Percorsi
coco_json_path = "./datasets/pianteInfestanti/train/_annotations.coco.json"  # Modifica con il percorso del tuo file JSON
output_dir = "./datasetConverted"  # Directory di output
images_dir = "./datasets/pianteInfestanti/train"  # Directory contenente le immagini originali

# Avvia la conversione
coco_to_yolo(coco_json_path, output_dir, images_dir)
