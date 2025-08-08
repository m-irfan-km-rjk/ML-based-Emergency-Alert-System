# 🚨 AI-Powered Emergency Detection & Alert System

An intelligent **emergency detection system** built using **PyTorch** (ResNet-18) and a Flask web frontend. Upload an image and the app classifies it into emergency categories (fire, accident, collapse, flood, or normal) and — for critical classes — can send SMS alerts through **Twilio** with a shortened Google Maps link to the estimated location.

---

## 🔍 What's included in this repo

* `app.py` — Flask web application that loads the trained PyTorch model, accepts image uploads, performs inference, fetches location by IP, shortens Google Maps links, and (optionally) sends Twilio SMS alerts.
* `templates/index.html` — Minimal front-end for uploading images and showing predictions (expected to exist in `templates/`).
* `static/` — For styles, JS and any static assets used by the front end.
* `best_emergency_model.pth` — Trained `ResNet-18` model file used by the Flask app (replace with your fine-tuned model file).

---

## ✅ Features (current)

* Single-file Flask app (`app.py`) that:

  * Loads a ResNet-18 model with a custom fully-connected head matching the repo's class set.
  * Accepts image uploads and returns a predicted label.
  * Uses `geocoder` to get location from the server's IP and builds a Google Maps link.
  * Shortens the Google Maps URL via TinyURL.
  * Sends SMS alerts through Twilio for mapped emergency classes.
  * Returns a base64-encoded version of the uploaded image for frontend preview.

---

## 📋 Class names (must match training)

The app expects the model to output one of these classes (case and ordering used in `app.py`):

```
['accident', 'collapse', 'fire', 'flood', 'Normal']
```

> If your trained model uses a different class order or names, update `class_names` in `app.py` and re-save the model accordingly.

---

## ☎️ Phone mapping for alerts

Only a subset of classes are mapped to phone numbers for SMS alerts. Edit `phone_mapping` in `app.py` with real phone numbers (E.164 format) before enabling Twilio.

Example mapping in `app.py`:

```py
phone_mapping = {
    "fire": "<+1234567890>",
    "accident": "<+1234567891>",
    "collapse": "<+1234567892>",
    "flood": "<+1234567893>
}
```

---

## 🔑 Twilio & Environment Variables

Set the following environment variables (do **not** hard-code secrets in source):

* `TWILIO_SID` — Your Twilio Account SID
* `TWILIO_AUTH` — Your Twilio Auth Token
* `TWILIO_MSG_SID` — Messaging service SID (or modify the code to use `from_` for a phone number)

You can export them in Linux/macOS like:

```bash
export TWILIO_SID="your_sid"
export TWILIO_AUTH="your_auth_token"
export TWILIO_MSG_SID="your_messaging_service_sid"
```

---

## 🧰 Dependencies

Install required packages (recommended in a virtualenv):

```bash
pip install flask torch torchvision pillow geocoder twilio requests
```

> If you use a GPU or a different torch build, install the matching `torch`/`torchvision` packages according to your CUDA version.

---

## 🚀 Running the App (development)

1. Ensure the trained model file `best_emergency_model.pth` is present in the project root (or update the path in `app.py`).
2. Set the Twilio environment variables if you plan to use SMS alerts.
3. Run the Flask app:

```bash
python app.py
```

4. Open `http://127.0.0.1:5000/` in your browser and upload an image.

---

## 🔧 Important Implementation Notes

* **Model shape & classes**: `app.py` constructs a `resnet18` and replaces `fc` with a `Linear` layer sized to `len(class_names)`. The model file must match this architecture and class ordering.
* **Location accuracy**: `geocoder.ip('me')` uses IP-based geolocation — it is approximate and often inaccurate for precise coordinates. For production, prefer device-supplied GPS or external metadata.
* **URL shortening**: The app calls TinyURL to shorten the Google Maps link. If rate limits are a concern, swap to a self-hosted or paid URL shortener.
* **Twilio limits**: Twilio may block or rate limit messages if the account is not configured correctly. For SMS sending, ensure the Twilio messaging service or `from` number is configured and verified for your destination numbers.
* **Safety checks**: The app currently sends SMS for any mapped label. Consider adding a confidence threshold (e.g. softmax score > 0.6) and human-in-the-loop confirmation before sending alerts.

---

## 🔒 Security & Privacy

* **Do not commit** Twilio credentials, personal phone numbers, or your model weights to a public repo. Use environment variables and `.gitignore`.
* **User IP & location** collected by the app may be sensitive. Ensure you have consent and follow local privacy laws when storing or transmitting location data.

---

## 🛠 Troubleshooting

* **Model load fails**: Check `torch` version compatibility and whether the model was saved with `state_dict()` vs full `torch.save(model)`; adjust loading accordingly.
* **`geocoder` returns `None`**: The machine may not have a public IP or the `geocoder` API could be blocked. Test manually by printing `g.latlng`.
* **Twilio exceptions**: Inspect the exception message returned when sending SMS — common issues include invalid credentials, unverified destination numbers, or missing messaging service.

---

## ♻️ Suggested Improvements for Future enhancements

* Add prediction confidence and only send alerts above a threshold.
* Support live video/CCTV streams (frame-sampling + inference pipeline).
* Add user/role authentication for the web UI and an audit log for sent alerts.
* Deploy behind a production-ready server (Gunicorn + nginx) and enable HTTPS.
* Reduce false alarms with multi-frame confirmation.
* Store incidents in a database for analytics and hotspot mapping.

Enable offline mode with queued alerts for poor connectivity.
---

## 📝 License

This project is released under the **MIT License**.

---

*If you want, I can also update the `index.html` template to show prediction confidence, a send/confirm SMS toggle, or add a simple Dockerfile for containerized deployment — tell me which you'd like.*
