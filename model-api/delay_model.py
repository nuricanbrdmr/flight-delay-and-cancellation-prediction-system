import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from google.colab import drive
import os
import warnings
import matplotlib.pyplot as plt
import joblib
warnings.filterwarnings('ignore')

# Google Drive'ı bağlama
drive.mount('/content/drive')

# Çalışma dizinini ayarlama
os.chdir('/content/drive/My Drive/Bitirme')

df = pd.read_csv(r"flights_Delay_final.csv")
df.head(5)

# İptal edilmiş uçuşları kaldırma
df = df[df['CANCELLED'] == 0]

# Tarih sütununu datetime'a çevirme ve yıl, ay, gün olarak ayırma
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], format='%Y-%m-%d')
df['YEAR'] = df['FL_DATE'].dt.year
df['MONTH'] = df['FL_DATE'].dt.month
df['DAY'] = df['FL_DATE'].dt.day

# Kategorik sütunlar için LabelEncoder
label_enc = LabelEncoder()
df['AIR'] = label_enc.fit_transform(df['AIRLINE_CODE'])
df['ORG'] = label_enc.fit_transform(df['ORIGIN_CITY'])
df['DST'] = label_enc.fit_transform(df['DEST_CITY'])

# Gereksiz sütunları silme
df.drop(['AIRLINE_CODE', 'ORIGIN_CITY', 'DEST_CITY', 'CANCELLED', 'FL_DATE', 'CANCELLATION_CODE', "CRS_ARR_TIME"], axis=1, inplace=True)

df.head()

# Veri Ön İşleme
# Eksik değerleri doldurma
print("Eksik değerler dolduruluyor...")
imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Ölçeklendirme (StandardScaler)
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Gecikme sürelerini sınıflara ayırma (Hedef değişken oluşturma)
df['DELAY_CLASS'] = df['DEP_DELAY'].apply(lambda x: 0 if x <= 0 else
                                         (1 if x <= 15 else
                                          (2 if x <= 30 else 3)))

# Sınıf dağılımını kontrol etme
print("\nSınıf dağılımı:")
print(df['DELAY_CLASS'].value_counts(normalize=True))

# Özellikleri ve hedefi hazırlama
y = df['DELAY_CLASS']
X = df.drop(['DELAY_CLASS', 'DEP_DELAY', "DEP_TIME"], axis=1)

# Verileri bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri Dengesizliğini Giderme (SMOTE)
print("Veri SMOTE ile Dengesizlik Gideriliyor...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train, y_train = smote.fit_resample(X_train, y_train)

df["DELAY_CLASS"].value_counts()

# SMOTE sonrası sınıf dağılımını kontrol etme
print("\nSınıf dağılımı:")
print(y_train.value_counts(normalize=True))

# Pipeline oluşturma
pipeline = ImbPipeline([
    ('classifier', XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        random_state=42,
        eval_metric='logloss'
    ))
])

# Modeli eğitme
print("\nModel eğitimi başlıyor...")
pipeline.fit(X_train, y_train)
print("Model eğitimi tamamlandı!")

# Model performansını değerlendirme
y_pred = pipeline.predict(X_test)

print("\nModel Performansı:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# Modeli ve gerekli bileşenleri kaydet
print("\nModel ve bileşenler kaydediliyor...")
joblib.dump(pipeline, 'model_delay.joblib')
joblib.dump(imputer, 'imputer_delay.joblib')
joblib.dump(scaler, 'scaler_delay.joblib')
joblib.dump(label_enc, 'label_enc_delay.joblib')

print("Kaydedilen dosyalar:")
print("✅ model_delay.joblib")
print("✅ imputer_delay.joblib")
print("✅ scaler_delay.joblib")
print("✅ label_enc_delay.joblib")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion matrix oluştur
cm = confusion_matrix(y_test, y_pred)

# Görselleştir
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1,2,3], yticklabels=[0, 1,2,3])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Eğitim ve test doğruluğunu karşılaştırma
pipeline.fit(X_train, y_train)

y_train_pred_delay = pipeline.predict(X_train)
train_accuracy_delay = accuracy_score(y_train, y_train_pred_delay)

y_test_pred_delay = pipeline.predict(X_test)
test_accuracy_delay = accuracy_score(y_test, y_test_pred_delay)

print(f"Gecikme Sınıfı - Eğitim Doğruluğu: {train_accuracy_delay:.2f}")
print(f"Gecikme Sınıfı - Test Doğruluğu: {test_accuracy_delay:.2f}")
print(f"Fark: {abs(train_accuracy_delay - test_accuracy_delay):.2f}")

# ==================== GELİŞMİŞ ANALİZ, XAI, SENARYO, HATA, İŞ ETKİSİ ====================
print("\n" + "="*60)
print("GELİŞMİŞ ANALİZ, XAI, SENARYO, HATA, İŞ ETKİSİ")
print("="*60)

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report

# 1. Özellik Önemleri (XAI)
feature_importance = pipeline.named_steps['classifier'].feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

print("\n🔍 EN ÖNEMLİ ÖZELLİKLER (Top 10):")
for i, row in feature_importance_df.head(10).iterrows():
    print(f"{row['feature']:<25} | {row['importance']:.4f}")

# 2. Senaryo Bazlı Performans Analizi (örnek senaryolar)
def scenario_performance_analysis_delay(X_test, y_test, y_pred, feature_names):
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['actual'] = y_test.values
    test_df['predicted'] = y_pred
    scenarios = {
        'Kötü Hava': test_df['prcp'] > test_df['prcp'].quantile(0.8),
        'İyi Hava': test_df['prcp'] < test_df['prcp'].quantile(0.2),
        'Yaz': test_df['MONTH'].isin([6,7,8]),
        'Kış': test_df['MONTH'].isin([12,1,2]),
        'Uzun Mesafe': test_df['DISTANCE'] > test_df['DISTANCE'].quantile(0.8),
        'Kısa Mesafe': test_df['DISTANCE'] < test_df['DISTANCE'].quantile(0.2),
        'Yoğun Saatler': test_df['CRS_DEP_TIME'].apply(lambda x: (x // 100) in [7,8,9,17,18,19]),
        'Gece': test_df['CRS_DEP_TIME'].apply(lambda x: (x // 100) in [0,1,2,3,4,5])
    }
    print(f"{'Senaryo':<20} {'Veri':<8} {'Accuracy':<10} {'F1(macro)':<10} {'Güçlü/Zayıf'}")
    print("-" * 60)
    for scenario_name, mask in scenarios.items():
        if mask.sum() > 10:
            scenario_actual = test_df.loc[mask, 'actual']
            scenario_pred = test_df.loc[mask, 'predicted']
            acc = accuracy_score(scenario_actual, scenario_pred)
            f1 = f1_score(scenario_actual, scenario_pred, average='macro', zero_division=0)
            perf = (acc + f1) / 2
            strength = "🟢 Güçlü" if perf > 0.7 else ("🟡 Orta" if perf > 0.5 else "🔴 Zayıf")
            print(f"{scenario_name:<20} {mask.sum():<8} {acc:<10.3f} {f1:<10.3f} {strength}")

scenario_performance_analysis_delay(X_test, y_test, y_pred, X_test.columns)

# 3. Hata Analizi (False Positives/Negatives)
def error_analysis_delay(y_true, y_pred):
    print("\n🔍 HATA ANALİZİ (Çoklu Sınıf):")
    report = classification_report(y_true, y_pred, output_dict=True)
    for cls in sorted(set(y_true)):
        print(f"Sınıf {cls}: Precision={report[str(cls)]['precision']:.2f}, Recall={report[str(cls)]['recall']:.2f}, F1={report[str(cls)]['f1-score']:.2f}")
    print("\nYanlış sınıflandırılan örneklerin dağılımı:")
    cm = confusion_matrix(y_true, y_pred)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                print(f"Gerçek: {i}, Tahmin: {j} => {cm[i, j]} örnek")

error_analysis_delay(y_test, y_pred)

# 4. İyileştirme Önerileri
print("\n💡 İYİLEŞTİRME ÖNERİLERİ:")
print("- Özellik mühendisliği ile hava durumu, saat, mesafe gibi etkili değişkenleri daha iyi modelle.")
print("- Sınıf dengesizliği için SMOTE parametrelerini optimize et.")
print("- Model parametrelerini (max_depth, learning_rate) grid search ile ayarla.")
print("- Ensemble veya farklı algoritmalar (LightGBM, CatBoost) dene.")
print("- Gecikme sınıflarını birleştirerek (ör. 0 vs 1-2-3) binary analiz de yapabilirsin.")

# 5. İş Etkisi ve ROI Analizi (örnek)
def business_impact_analysis_delay(cm):
    # Sınıf 0: zamanında, 1-2-3: gecikmeli
    on_time = cm[0,0]
    late_predicted_on_time = cm[1:,0].sum()
    on_time_predicted_late = cm[0,1:].sum()
    late_correct = cm[1:,1:].sum() - np.trace(cm[1:,1:])
    # Varsayımsal maliyetler
    COST_LATE_UNDETECTED = 200
    COST_FALSE_ALARM = 50
    REVENUE_CORRECT_LATE = 100
    REVENUE_ON_TIME = 10
    total_cost = late_predicted_on_time * COST_LATE_UNDETECTED + on_time_predicted_late * COST_FALSE_ALARM
    total_revenue = on_time * REVENUE_ON_TIME + late_correct * REVENUE_CORRECT_LATE
    net_impact = total_revenue - total_cost
    print(f"\n💰 İş Etkisi Analizi:")
    print(f"  - Zamanında doğru tahmin: {on_time} (Gelir: ${on_time*REVENUE_ON_TIME})")
    print(f"  - Gecikmeli doğru tahmin: {late_correct} (Gelir: ${late_correct*REVENUE_CORRECT_LATE})")
    print(f"  - Gecikmeli olup zamanında tahmin edilen: {late_predicted_on_time} (Maliyet: ${late_predicted_on_time*COST_LATE_UNDETECTED})")
    print(f"  - Zamanında olup gecikmeli tahmin edilen: {on_time_predicted_late} (Maliyet: ${on_time_predicted_late*COST_FALSE_ALARM})")
    print(f"  - Net Etki: ${net_impact}")

cm = confusion_matrix(y_test, y_pred)
business_impact_analysis_delay(cm)

# 6. Executive Summary
print("\n" + "="*60)
print("EXECUTIVE SUMMARY - YÖNETİCİ ÖZETİ")
print("="*60)
print(f"Genel Doğruluk: {accuracy_score(y_test, y_pred):.2%}")
print(f"Makro F1 Skoru: {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"En önemli özellikler: {', '.join(feature_importance_df.head(3)['feature'])}")
print(f"Senaryo bazlı analizde en güçlü olduğu alanlar: (ör. Yaz, Kötü Hava, Uzun Mesafe)")
print(f"Yanlış tahminlerin etkisi: Gecikmeli uçuşları zamanında tahmin etmek müşteri memnuniyetsizliğine ve tazminat maliyetine yol açar. Zamanında uçuşları gecikmeli tahmin etmek ise gereksiz alarm ve operasyonel karmaşa yaratır.")
print(f"İyileştirme için: Özellik mühendisliği, model parametre optimizasyonu ve daha fazla veri önerilir.")

# Özellik önemlerini görselleştirme
importances = pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Gecikme Sınıflandırması - Özellik Önem Dereceleri", fontsize=14, pad=20)
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=45, ha='right')
plt.xlabel('Özellikler')
plt.ylabel('Önem Derecesi')
plt.tight_layout()
plt.show()

# Örnek tahmin için yeni veri
new_data = pd.DataFrame({
    'YEAR': [2024],
    'MONTH': [5],
    'DAY': [15],
    'AIR': [14],
    'ORG': [289],
    'DST': [202],
    'CRS_DEP_TIME': [2155],
    'DISTANCE': [612],
    'tmin': [18.9],
    'tmax': [25],
    'prcp': [0],
    'snow': [0],
    'wdir': [312],
    'wspd': [11.5],
    'wpgt': [0],
    'pres': [1018.6],
    'tsun': [0]
}, index=[0])

# Eksik sütunları kontrol etme ve ekleme
missing_columns = set(X_train.columns) - set(new_data.columns)
for col in missing_columns:
    new_data[col] = 0

# Sütun sırasını train verisi ile aynı yapma
new_data = new_data[X_train.columns]

# Tahmin yapma
prediction = pipeline.predict(new_data)
prediction_proba = pipeline.predict_proba(new_data)

# Gecikme sınıfları sözlüğü
delay_classes = {
    0: "Zamanında veya erken",
    1: "Hafif gecikme (1-15 dakika)",
    2: "Orta gecikme (16-30 dakika)",
    3: "Ciddi gecikme (30+ dakika)"
}

# Tahmin sonuçlarını yazdırma
print("\nTahmin Sonucu:")
print(f"Tahmin edilen sınıf: {delay_classes[prediction[0]]}")
print("\nSınıf olasılıkları:")
for i, prob in enumerate(prediction_proba[0]):
    print(f"{delay_classes[i]}: {prob:.2%}")

# ==================== VİZUALİZASYON VE RAPORLAMA ====================
print("\n" + "="*60)
print("GÖRSELLEŞTİRME VE RAPORLAMA")
print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np

# 1. Feature Importance
plt.figure(figsize=(8, 6))
plt.barh(feature_importance_df.head(10)['feature'], feature_importance_df.head(10)['importance'])
plt.title('En Önemli 10 Özellik')
plt.xlabel('Önem Skoru')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 3. ROC Curve (One-vs-Rest)
n_classes = len(np.unique(y_test))
y_test_bin = label_binarize(y_test, classes=[0,1,2,3])
y_score = pipeline.predict_proba(X_test)
plt.figure(figsize=(8,6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Sınıf {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Precision-Recall Curve (One-vs-Rest)
plt.figure(figsize=(8,6))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'Sınıf {i} (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (One-vs-Rest)')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Senaryo Bazlı F1 Bar Plot
# Senaryo analizini tekrar çalıştırıp F1 skorlarını topluyoruz
def scenario_f1s(X_test, y_test, y_pred, feature_names):
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['actual'] = y_test.values
    test_df['predicted'] = y_pred
    scenarios = {
        'Kötü Hava': test_df['prcp'] > test_df['prcp'].quantile(0.8),
        'İyi Hava': test_df['prcp'] < test_df['prcp'].quantile(0.2),
        'Yaz': test_df['MONTH'].isin([6,7,8]),
        'Kış': test_df['MONTH'].isin([12,1,2]),
        'Uzun Mesafe': test_df['DISTANCE'] > test_df['DISTANCE'].quantile(0.8),
        'Kısa Mesafe': test_df['DISTANCE'] < test_df['DISTANCE'].quantile(0.2),
        'Yoğun Saatler': test_df['CRS_DEP_TIME'].apply(lambda x: (x // 100) in [7,8,9,17,18,19]),
        'Gece': test_df['CRS_DEP_TIME'].apply(lambda x: (x // 100) in [0,1,2,3,4,5])
    }
    scenario_names = []
    scenario_f1s = []
    for scenario_name, mask in scenarios.items():
        if mask.sum() > 10:
            scenario_actual = test_df.loc[mask, 'actual']
            scenario_pred = test_df.loc[mask, 'predicted']
            f1 = f1_score(scenario_actual, scenario_pred, average='macro', zero_division=0)
            scenario_names.append(scenario_name)
            scenario_f1s.append(f1)
    return scenario_names, scenario_f1s

names, f1s = scenario_f1s(X_test, y_test, y_pred, X_test.columns)
plt.figure(figsize=(10,5))
plt.bar(names, f1s)
plt.ylabel('F1 Score (macro)')
plt.title('Senaryolara Göre F1 Skoru')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()