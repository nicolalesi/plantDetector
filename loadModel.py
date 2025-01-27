import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import os

# Load the model
model = tf.keras.models.load_model('./modelliGenerati/mergedData_model.h5')
class_names = ['crop', 'weed'] 

def process_image(img_path):
    """
    Preprocesses an image for model prediction.

    Args:
        img_path: Path to the image file.

    Returns:
        A NumPy array representing the preprocessed image.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(139, 139))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.image.resize(img_array, (139, 139))
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

def generate_heatmap(img_array):
    """
    Generates a heatmap for the predicted class.

    Args:
        img_array: A NumPy array representing the preprocessed image.

    Returns:
        A tuple containing the heatmap and the predicted class index.
    """
    predicted_class = np.argmax(model.predict(img_array)[0])
    last_conv_layer = model.get_layer('mixed10') 
    heatmap_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = heatmap_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)

    heatmap_resized = cv2.resize(heatmap, (139, 139))
    return heatmap_resized, predicted_class

def load_and_predict(file_path):
    """
    Loads an image, performs prediction and heatmap generation,
    and updates the GUI elements.

    Args:
        file_path: Path to the image file.
    """
    if not file_path:
        return

    try:
        # Process the image
        img_array = process_image(file_path) 
        heatmap, predicted_class = generate_heatmap(img_array)

        # Update label text
        label_result.config(text=f"Predizione: {class_names[predicted_class]}")

        # Load and display original image
        original_image = Image.open(file_path).resize((300, 300))
        original_photo = ImageTk.PhotoImage(original_image)
        label_original.config(image=original_photo)
        label_original.image = original_photo

        # Generate and display heatmap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_image = Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        heatmap_resized = heatmap_image.resize((300, 300))
        heatmap_photo = ImageTk.PhotoImage(heatmap_resized)
        label_heatmap.config(image=heatmap_photo)
        label_heatmap.image = heatmap_photo

    except Exception as e:
        messagebox.showerror("Errore", f"Si è verificato un errore: {e}")

# Create the main window
window = tk.Tk()
window.title("Rilevamento Malattie delle Piante")
window.geometry("800x600")

# Button to load image
btn_load = tk.Button(window, text="Carica Immagine", command=lambda: load_and_predict(filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])))
btn_load.pack(pady=20)

# Label to display prediction result
label_result = tk.Label(window, text="Predizione: ", font=("Arial", 16))
label_result.pack(pady=10)

# Frame for images
frame_images = tk.Frame(window)
frame_images.pack(pady=20)

# Labels to display original image and heatmap
label_original = tk.Label(frame_images, text="Immagine Originale")
label_original.pack(side=tk.LEFT, padx=20)
label_heatmap = tk.Label(frame_images, text="Heatmap")
label_heatmap.pack(side=tk.RIGHT, padx=20)

# Start the GUI event loop
window.mainloop()