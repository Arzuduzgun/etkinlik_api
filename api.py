from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("multi_model.pkl")
encoders = joblib.load("encoders.pkl")
mlb = joblib.load("mlb.pkl")

@app.route("/oneri", methods=["POST"])
def oneri():
    veri = request.json

    girdi = pd.DataFrame([[
        veri["dil"],
        veri["iletisim"],
        veri["motor"],
        veri["hassasiyet"],
        veri["ilgi"]
    ]], columns=["dil", "iletisim", "motor", "hassasiyet", "ilgi"])

    for col in girdi.columns:
        girdi[col] = encoders[col].transform(girdi[col])

    tahmin = model.predict(girdi)
    sonuc = mlb.inverse_transform(tahmin)[0]

    return jsonify({"oneriler": list(sonuc)})

if __name__ == "__main__":
    app.run(debug=True)
