from flask import Flask, render_template, request
import torch
import os
from PIL import Image
from torchvision import transforms, models
import geocoder
from twilio.rest import Client
import requests
from io import BytesIO
import base64

app = Flask(__name__)

# ===== 1. Define Class Names (MATCH training) =====
class_names = ['accident', 'collapse', 'fire', 'flood', 'Normal']

# ===== 2. Define Phone Mapping (Only critical classes) =====
phone_mapping = {
    "fire": "Number1",
    "accident": "Number2",
    "collapse": "Number3",
    "flood": "Number4",
}

# ===== 3. Twilio Credentials =====
TWILIO_SID = "__TWILIO_SID__"
TWILIO_AUTH = "__TWILIO_AUTH__"
TWILIO_MSG_SID = "__TWILIO_MSG_SID__"

# ===== 4. Load Model (ResNet18) with 7-class FC Layer =====
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("best_emergency_model.pth", map_location=torch.device("cpu")))
model.eval()

# ===== 5. Image Transform =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== 6. Location Fetching =====
def get_location():
    try:
        g = geocoder.ip('me')
        if g.ok:
            lat, lon = g.latlng
            return lat, lon, g.city, g.state, g.country
    except:
        pass
    return None, None, None, None, None

# ===== 7. TinyURL Shortener =====
def shorten_url(long_url):
    try:
        r = requests.get(f"http://tinyurl.com/api-create.php?url={long_url}")
        if r.status_code == 200:
            return r.text
    except:
        pass
    return long_url

# ===== 8. Flask Routes =====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return "No image uploaded!", 400

        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")

        # Base64 for frontend display
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode()

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]

        # Get location
        lat, lon, city, state, country = get_location()
        map_url = shorten_url(f"https://www.google.com/maps/search/?api=1&query={lat},{lon}") if lat and lon else "Unavailable"

        # Prepare message
        alert_message = f"🚨 '{predicted_class.upper()}' detected at {city}, {state}, {country}\nMap: {map_url}"
        to_phone = phone_mapping.get(predicted_class.lower())

        # Send SMS if class is mapped
        sms_status = "⚠️ No SMS mapping for this emergency type."
        if to_phone:
            try:
                client = Client(TWILIO_SID, TWILIO_AUTH)
                client.messages.create(
                    messaging_service_sid=TWILIO_MSG_SID,
                    to=to_phone,
                    body=alert_message
                )
                sms_status = f"✅ SMS sent to {to_phone}"
            except Exception as e:
                sms_status = f"❌ SMS failed: {str(e)}"

        return render_template("index.html",
                               prediction=predicted_class,
                               city=city, state=state, country=country,
                               map_url=map_url,
                               sms_status=sms_status,
                               image_data=image_data)

    return render_template("index.html")

# ===== 9. Run App =====
if __name__ == "__main__":
    app.run(debug=True)