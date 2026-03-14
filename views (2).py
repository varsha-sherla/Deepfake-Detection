from django.shortcuts import render
import cv2
import numpy as np
import tensorflow as tf
from django.core.files.storage import default_storage
from django.conf import settings
import os
from uuid import uuid4

# Load the trained model
model = tf.keras.models.load_model("xception_deepfake_image.h5")


def generate_gradcam(img_array, original_img_path, model):
    """Generate Grad-CAM heatmap overlay for explainability."""
    last_conv_layer = model.get_layer(index=-5)
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    # Load and resize original image
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))

    # Resize and apply colormap to heatmap
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save heatmap to media directory
    heatmap_filename = f"heatmaps/heatmap_{uuid4().hex}.jpg"
    heatmap_path = os.path.join(settings.MEDIA_ROOT, heatmap_filename)
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    cv2.imwrite(heatmap_path, superimposed_img)

    return settings.MEDIA_URL + heatmap_filename


def detect_deepfake(request):
    """Handle image upload and return deepfake prediction with Grad-CAM."""
    if request.method == "POST" and request.FILES.get("image"):
        file = request.FILES["image"]
        filename = f"uploads/{uuid4().hex}_{file.name}"
        file_path = default_storage.save(filename, file)
        temp_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        # Read and preprocess image
        image = cv2.imread(temp_file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (224, 224)) / 255.0
        input_array = np.expand_dims(resized, axis=0)

        # Predict
        prob = model.predict(input_array)[0][0]
        label = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100

        # Generate Grad-CAM heatmap
        heatmap_url = generate_gradcam(input_array, temp_file_path, model)

        return render(request, "detection/result.html", {
            "label": label,
            "confidence": round(confidence, 2),
            "original": settings.MEDIA_URL + file_path,
            "heatmap": heatmap_url,
        })

    return render(request, "detection/upload.html")
