from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load model klasifikasi hewan
model = tf.keras.models.load_model('model/animal_classification_coba1.h5')

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Endpoint untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah file gambar ada dalam permintaan
    if 'image' not in request.files:
        return jsonify({"error": "Tidak ada file gambar pada permintaan"}), 400
    
    file = request.files['image']
    
    # Load dan preprocess gambar
    try:
        # Buka gambar menggunakan PIL
        img = Image.open(file)
        # Ubah ukuran gambar menjadi 150x150 piksel
        img = img.resize((150, 150))
        # Konversi gambar ke array numpy dengan dimensi (150,150,3)
        img_array = np.array(img)
        
        # Pastikan gambar memiliki 3 channel (RGB)
        if img_array.shape[-1] != 3:
            img = img.convert('RGB')
            img_array = np.array(img)
        
        # Tambah dimensi batch menjadi (1,150,150,3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalisasi data gambar (jika dibutuhkan oleh model)
        img_array = img_array / 255.0
        
        # Melakukan prediksi
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Mapping hasil prediksi ke label kelas
        class_names = {0: "Kucing", 1: "Anjing", 2: "Ular"}
        result = {
            "prediction": class_names.get(predicted_class, "Tidak Diketahui"),
            "confidence": float(np.max(predictions))  # Confidence dari prediksi
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
