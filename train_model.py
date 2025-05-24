import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
import joblib

data = pd.DataFrame([
    ["hiç konuşmuyor", "göz teması kurmuyor", "çok zayıf", "ses", "hayvanlar", ["hayvan seslerini taklit etme", "göz teması kurmayı ödüllendirme", "parmak boyası ile oyun"]],
    ["hiç konuşmuyor", "az kuruyor", "orta", "dokunma", "resim", ["hayvan seslerini taklit etme", "sırayla bakışma oyunu", "oyun hamuru", "resim çizme"]],
    ["tek kelime", "az kuruyor", "orta", "ışık", "sayılar", ["nesne ve isim eşleştirme", "sırayla bakışma oyunu", "mandal takma", "nesneleri sayma"]],
    ["iki kelime", "aktif", "iyi", "dokunma", "kitap", ["rastgele hikayeyi anlatma", "farklı dokulu nesneleri keşfetme"]],
    ["akıcı", "aktif", "iyi", "ses", "müzik", ["meslek taklidi", "ritim tutma", "yazı çalışmaları"]],
    ["tek kelime", "göz teması kurmuyor", "orta", "ışık", "taşıtlar", ["göz teması kurmayı ödüllendirme", "taşıt adını söyleme"]],
    ["hiç konuşmuyor", "az kuruyor", "çok zayıf", "ses", "müzik", ["hayvan seslerini taklit etme", "hafif sesli müzik dinleme"]],
    ["akıcı", "aktif", "iyi", "dokunma", "resim", ["meslek taklidi", "farklı dokulu nesneleri keşfetme", "resim çizme"]],
    ["iki kelime", "az kuruyor", "orta", "ışık", "puzzle", ["rastgele hikayeyi anlatma", "oyun hamuru", "az parçalı puzzle tamamlama"]],
    ["tek kelime", "göz teması kurmuyor", "orta", "ses", "hayvanlar", ["nesne ve isim eşleştirme", "göz teması kurmayı ödüllendirme"]],
    ["akıcı", "aktif", "iyi", "ses", "kitap", ["meslek taklidi", "hafif sesli müzik dinleme", "hikaye anlatma"]],
    ["iki kelime", "az kuruyor", "iyi", "dokunma", "resim", ["rastgele hikayeyi anlatma", "resim çizme"]],
    ["tek kelime", "az kuruyor", "orta", "ışık", "kitap", ["nesne ve isim eşleştirme", "sırayla bakışma oyunu", "hikaye anlatma"]],
    ["hiç konuşmuyor", "göz teması kurmuyor", "çok zayıf", "dokunma", "puzzle", ["hayvan seslerini taklit etme", "göz teması kurmayı ödüllendirme", "parmak boyası", "az parçalı puzzle"]],
    ["iki kelime", "aktif", "iyi", "ışık", "sayılar", ["rastgele hikayeyi anlatma", "nesneleri sayma"]],
    ["tek kelime", "az kuruyor", "orta", "ışık", "resim", ["oyun hamuru", "resim çizme"]],
    ["hiç konuşmuyor", "az kuruyor", "orta", "dokunma", "hayvanlar", ["hayvan seslerini taklit etme", "oyun hamuru", "dokulu nesne keşfi"]],
    ["akıcı", "aktif", "iyi", "dokunma", "taşıtlar", ["meslek taklidi", "taşıt adını söyleme"]],
    ["iki kelime", "az kuruyor", "orta", "ses", "müzik", ["rastgele hikaye", "ritim tutma"]],
    ["tek kelime", "az kuruyor", "iyi", "dokunma", "puzzle", ["oyun hamuru", "az parçalı puzzle"]],
], columns=["dil", "iletisim", "motor", "hassasiyet", "ilgi", "oneriler"])


# Özellikleri etiketle
encoders = {}
for col in ["dil", "iletisim", "motor", "hassasiyet", "ilgi"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Girdi ve çıktı
X = data[["dil", "iletisim", "motor", "hassasiyet", "ilgi"]]
y = data["oneriler"]

# Çoklu etiketleri binary formata çevir
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)

# Model oluştur
model = OneVsRestClassifier(DecisionTreeClassifier())
model.fit(X, Y)

# Model ve kodlayıcıları kaydet
joblib.dump(model, "multi_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(mlb, "mlb.pkl")

print("✅ Eğitim tamamlandı. Model ve yardımcılar kaydedildi.")