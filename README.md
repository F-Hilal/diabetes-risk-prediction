# Diabetes Risk Prediction (Deep Learning + Explainability)

Bu projede, **Pima Indians Diabetes** veri setini kullanarak bireylerin diyabet olma riskini tahmin eden bir **yapay sinir ağı (MLP)** modeli geliştirdim. Ayrıca modelin kararlarını açıklamak için **SHAP** ile yorumlanabilirlik analizi yaptım.

---

**Amaç**

- Diyabet hastalığı riskini tahmin etmek  
- Hangi özelliklerin (glikoz, BMI, yaş vb.) tahminde daha etkili olduğunu görmek  
- Kurulan modelin kararlarını sadece "doğru/yanlış" değil, **“neden böyle tahmin yaptı?”** sorusuna cevap verecek şekilde incelemek  

---

**Veri Hazırlama**

Projede yapılan başlıca veri ön işleme adımları:

- Veri setini `pandas` ile okuma  
- Biyolojik olarak 0 olamayacak değerleri (Glucose, BloodPressure, SkinThickness, Insulin, BMI) **eksik (NaN)** olarak işaretleme  
- Bu eksik değerleri **medyan ile doldurma**  
- Özellik ve hedef değişkenleri (X, y) ayırma  
- Veriyi eğitim (%80) ve test (%20) olarak bölme  
- `StandardScaler` ile tüm özellikleri ölçeklendirme  

Bu adımlar, modelin daha sağlıklı öğrenmesi ve dengesiz/hatalı veriden etkilenmemesi için uygulandı.

---

**Model: Yapay Sinir Ağı (MLP)**

Model, **TensorFlow / Keras** ile kuruldu:

- Giriş katmanı: 8 özellik  
- Gizli katmanlar:
  - 64 nöron, ReLU aktivasyonu  
  - 32 nöron, ReLU aktivasyonu  
  - Aralarda **Dropout** katmanları (overfitting’i azaltmak için)  
- Çıkış katmanı: 1 nöron, sigmoid aktivasyonu (diyabet olma olasılığı)

Eğitim:

- Loss: `binary_crossentropy`  
- Optimizer: `adam`  
- Metrik: `accuracy`  
- Sınıf dengesizliğini azaltmak için `class_weight` kullanıldı.  

---

**Model Performansı**

Test verisi üzerinde:

- **Doğruluk (Accuracy):** ~%73–75  
- **AUC (ROC Eğrisi Altındaki Alan):** ~0.80  
- Confusion matrix ve sınıflandırma raporu (`precision`, `recall`, `f1-score`) ile daha detaylı performans analizi yapıldı.

Model, rastgele tahmin eden bir modele göre çok daha iyi performans gösteriyor ve diyabetli bireyleri makul bir başarıyla ayırt edebiliyor.

---

**Model Yorumlama: SHAP Analizi**

Modelin kararlarını açıklamak için **SHAP (SHapley Additive exPlanations)** kullandım.

SHAP ile elde edilen bulgular:

- **Glucose (Kan şekeri):** Modelin en önemli özelliği. Glikoz seviyesi arttıkça diyabet tahmini belirgin şekilde artıyor.  
- **BMI:** İkinci en önemli özellik. Yüksek BMI, diyabet riskini artırıyor.  
- **Pregnancies, DiabetesPedigreeFunction, Age:** Orta düzeyde etkiye sahip.  
- **BloodPressure ve SkinThickness:** Görece daha düşük etkiye sahip.

Bu sonuçlar, tıbbi literatürle uyumludur: yüksek glikoz ve BMI, diyabet riskinin başlıca göstergeleridir.

---

**Kullanılan Teknolojiler**

- **Python**  
- **Pandas, NumPy**  
- **Scikit-learn**  
- **TensorFlow / Keras**  
- **Matplotlib, Seaborn**  
- **SHAP**

---

## Projeyi Çalıştırma

Gerekli kütüphaneleri yükledikten sonra:

```bash
python diabetes_risk_prediction.py
