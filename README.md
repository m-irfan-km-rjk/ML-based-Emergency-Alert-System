# 🚨 AI-Powered Emergency Detection & Alert System

An intelligent **emergency detection system** built using **PyTorch** and **ResNet-18** for real-time classification of emergency situations from images.  
The system can detect multiple emergency categories (e.g., fire, accidents, building collapse, and normal situations) and can be integrated with **Twilio** for automated alert notifications.

---

## 📌 Features
- **Deep Learning Model**: Uses a pretrained **ResNet-18** model fine-tuned for emergency classification.
- **Image Preprocessing & Augmentation**: Improves model generalization with techniques like resizing, cropping, rotation, color jitter, and normalization.
- **High Accuracy**: Trains with validation monitoring and saves the best-performing model.
- **Real-Time Detection**: Allows testing of custom images for prediction.
- **Alert System Integration** *(Optional)*: Can send automated alerts via **Twilio SMS API** when an emergency is detected.

---

## 📂 Dataset
- The dataset should be structured into subfolders, one for each class:
```
emergency_dataset/
│── Fire/
│── Accident/
│── Collapse/
│── Normal/
```
- Images in each folder represent that class.
- The dataset is automatically split into **80% training** and **20% validation**.

---

## 🛠️ Technologies Used
- **Python 3**
- **PyTorch**
- **Torchvision**
- **PIL (Pillow)**
- **Matplotlib**
- **Twilio API** *(for SMS alerts)*

---

## 🚀 How It Works
1. **Data Loading** – Images are loaded from the dataset directory.
2. **Training** – ResNet-18 is fine-tuned with data augmentation.
3. **Validation** – Accuracy is checked after each epoch, and the best model is saved.
4. **Prediction** – The trained model predicts the emergency category for a given image.
5. **Alert (Optional)** – Sends SMS notifications if an emergency is detected.

---

## 📷 Example Usage
```bash
$ python emergency_detection.py
Enter the full path to the image you want to test: /path/to/image.jpg
🧾 Predicted Class: Fire
```
*(If Twilio integration is enabled, an alert SMS will be sent automatically.)*

---

## 🔮 Future Improvements
- Integrate with **CCTV live feed** for real-time detection.
- Expand dataset for more emergency categories.
- Deploy as a **web app** for broader accessibility.
- Add **audio alerts** and **email notifications**.

---

## 📜 License
This project is open-source and available under the **MIT License**.

---
💡 *Developed with a mission to improve emergency response through AI.*
