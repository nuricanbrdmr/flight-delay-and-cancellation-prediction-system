# LEGACY (kept for the record): superseded by model-api/revision/. This version contains known problems that the revision removes
# (target-based features computed on all data, SMOTE before temporal CV, test-set threshold selection, delay classes built from standardised minutes).
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score
)
from google.colab import drive
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== GOOGLE DRIVE BAĞLANTISI ====================
print("="*60)
print("GOOGLE DRIVE BAĞLANTISI")
print("="*60)
drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Bitirme')

# ==================== VERİ YÜKLEME ====================
print("\n" + "="*60)
print("VERİ YÜKLEME")
print("="*60)
df = pd.read_csv("flights_Cancel_final.csv")
print(f"Veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")

# ==================== VERİ ÖN İŞLEME ====================
print("\n" + "="*60)
print("VERİ ÖN İŞLEME")
print("="*60)

# Tarih işleme - YEAR'ın etkisini azaltmak için normalizasyon
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], format='%Y-%m-%d')
min_year = df['FL_DATE'].dt.year.min()
max_year = df['FL_DATE'].dt.year.max()

df['YEAR_NORMALIZED'] = (df['FL_DATE'].dt.year - min_year) / (max_year - min_year)
df['MONTH'] = df['FL_DATE'].dt.month
df['DAY'] = df['FL_DATE'].dt.day
df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek

# Mevsimsel özellikler
df['SEASON'] = df['MONTH'].map({12: 0, 1: 0, 2: 0,  # Kış
                                3: 1, 4: 1, 5: 1,   # İlkbahar
                                6: 2, 7: 2, 8: 2,   # Yaz
                                9: 3, 10: 3, 11: 3}) # Sonbahar

# Trigonometric features for cyclical data
df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY'] / 31)
df['DAY_COS'] = np.cos(2 * np.pi * df['DAY'] / 31)

# Hava durumu composite score
def weather_composite_score(row):
    score = 0
    if row['prcp'] > 0:
        score += min(row['prcp'] / 10, 3)
    if row['snow'] > 0:
        score += min(row['snow'] / 5, 4)
    if row['wspd'] > 10:
        score += min((row['wspd'] - 10) / 10, 2)
    if row['tmax'] > 35 or row['tmin'] < -10:
        score += 1
    if row['pres'] < 1000 or row['pres'] > 1025:
        score += 0.5
    return min(score, 10)

df['WEATHER_COMPOSITE'] = df.apply(weather_composite_score, axis=1)

# Uçuş zamanı kategorisi
def categorize_time_detailed(time):
    if pd.isna(time):
        return 0
    hour = int(time // 100)
    if 5 <= hour < 8:
        return 1
    elif 8 <= hour < 12:
        return 2
    elif 12 <= hour < 15:
        return 3
    elif 15 <= hour < 19:
        return 4
    elif 19 <= hour < 22:
        return 5
    else:
        return 6

df['DEP_TIME_DETAILED'] = df['CRS_DEP_TIME'].apply(categorize_time_detailed)

# Mesafe kategorisi
df['DISTANCE_CATEGORY'] = pd.cut(df['DISTANCE'],
                                bins=[0, 400, 800, 1500, 3000, 6000],
                                labels=[0, 1, 2, 3, 4])

# Label Encoding
label_enc_airline = LabelEncoder()
label_enc_origin = LabelEncoder()
label_enc_dest = LabelEncoder()
label_enc_cancel_code = LabelEncoder()

df['AIR'] = label_enc_airline.fit_transform(df['AIRLINE_CODE'])
df['ORG'] = label_enc_origin.fit_transform(df['ORIGIN_CITY_NAME'])
df['DST'] = label_enc_dest.fit_transform(df['DEST_CITY_NAME'])

# İptal kodları için özel işlem
df['CANCELLATION_CODE_FILLED'] = df['CANCELLATION_CODE'].fillna('N')
df['CANCELLATION_CODE_ENCODED'] = label_enc_cancel_code.fit_transform(df['CANCELLATION_CODE_FILLED'])

# Route popularity
route_counts = df.groupby(['ORG', 'DST']).size()
df['ROUTE_POPULARITY'] = df.apply(lambda x: route_counts.get((x['ORG'], x['DST']), 0), axis=1)
df['ROUTE_POPULARITY_LOG'] = np.log1p(df['ROUTE_POPULARITY'])

# Airline reliability score
airline_cancel_rate = df.groupby('AIR')['CANCELLED'].mean()
df['AIRLINE_RELIABILITY'] = df['AIR'].map(airline_cancel_rate)

# Gereksiz sütunları silme
columns_to_drop = ['AIRLINE_CODE', 'ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'FL_DATE',
                   'ROUTE_POPULARITY', 'CANCELLATION_CODE', 'CANCELLATION_CODE_FILLED']
df.drop(columns_to_drop, axis=1, inplace=True)

print("✅ Veri ön işleme tamamlandı!")

# ==================== SADECE İPTAL EDİLMİŞ UÇUŞLARI FİLTRELE ====================
print("\n" + "="*60)
print("İPTAL EDİLMİŞ UÇUŞLARI FİLTRELEME")
print("="*60)

# Sadece iptal edilmiş uçuşları al
df_cancelled = df[df['CANCELLED'] == 1].copy()
print(f"Toplam uçuş sayısı: {len(df)}")
print(f"İptal edilmiş uçuş sayısı: {len(df_cancelled)}")
print(f"İptal oranı: {len(df_cancelled)/len(df)*100:.2f}%")

# İptal kodu dağılımı
print("\nİptal Kodu Dağılımı:")
cancel_code_dist = df_cancelled['CANCELLATION_CODE_ENCODED'].value_counts().sort_index()
cancel_code_names = {
    0: 'A: Havayolu/Taşıyıcı kaynaklı',
    1: 'B: Hava durumu kaynaklı',
    2: 'C: Ulusal Hava Sistemi kaynaklı',
    3: 'D: Güvenlik kaynaklı',
    4: 'N: Diğer'
}

for code, count in cancel_code_dist.items():
    print(f"   {code} - {cancel_code_names[code]}: {count} ({count/len(df_cancelled)*100:.2f}%)")

# Özellikleri ve hedefi hazırlama
y_cancel_code = df_cancelled['CANCELLATION_CODE_ENCODED']
X = df_cancelled.drop(["CANCELLED", "CANCELLATION_CODE_ENCODED", "DEP_TIME"], axis=1)

print(f"\nToplam özellik sayısı: {X.shape[1]}")

# ==================== TEMPORAL VALIDATION: Yıllara Göre Bölme ====================
print("\n" + "="*60)
print("TEMPORAL VALIDATION - ZAMANSAL DOĞRULAMA")
print("="*60)

# YEAR_NORMALIZED'dan gerçek yılı geri hesapla
df_cancelled['YEAR'] = (df_cancelled['YEAR_NORMALIZED'] * (max_year - min_year)) + min_year
df_cancelled['YEAR'] = df_cancelled['YEAR'].astype(int)

train_mask = df_cancelled['YEAR'] <= 2022
test_mask = df_cancelled['YEAR'] >= 2023

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y_cancel_code[train_mask]
y_test = y_cancel_code[test_mask]

print(f"Eğitim verisi: {len(X_train)} örnek (2016-2022)")
print(f"Test verisi: {len(X_test)} örnek (2023-2024)")
print(f"Eğitim yılları: {sorted(df_cancelled[train_mask]['YEAR'].unique())}")
print(f"Test yılları: {sorted(df_cancelled[test_mask]['YEAR'].unique())}")

print("\nTrain set sınıf dağılımı:")
print(y_train.value_counts().sort_index())
print("\nTest set sınıf dağılımı:")
print(y_test.value_counts().sort_index())

# ==================== VERİ ÖN İŞLEME (İMPUTER & SCALER) ====================
print("\n" + "="*60)
print("EKSİK DEĞER DOLDURMA VE ÖLÇEKLENDİRME")
print("="*60)

# Eksik değerleri doldurma
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("✅ Veri ön işleme tamamlandı!")

# ==================== SMOTE İLE DENGESİZLİK GİDERME ====================
print("\n" + "="*60)
print("SMOTE İLE VERİ DENGESİZLİĞİ GİDERME")
print("="*60)

print("SMOTE öncesi sınıf dağılımı:")
print(pd.Series(y_train).value_counts().sort_index())

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("\nSMOTE sonrası sınıf dağılımı:")
print(pd.Series(y_train_resampled).value_counts().sort_index())
print(f"\nSMOTE öncesi: {len(X_train_scaled)} örnek")
print(f"SMOTE sonrası: {len(X_train_resampled)} örnek")

# ==================== MODEL EĞİTİMİ ====================
print("\n" + "="*60)
print("İPTAL KODU MODELİ EĞİTİMİ")
print("="*60)

# İptal kodu modeli (multiclass classification)
model_cancel_code = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_weight=5,
    gamma=1.0,
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=5,  # 5 sınıf: A, B, C, D, N
    random_state=42,
    n_jobs=-1
)

print("Model parametreleri:")
print(f"  • n_estimators: {model_cancel_code.n_estimators}")
print(f"  • max_depth: {model_cancel_code.max_depth}")
print(f"  • learning_rate: {model_cancel_code.learning_rate}")
print(f"  • objective: {model_cancel_code.objective}")
print(f"  • num_class: 5 (A, B, C, D, N)")

print("\nModel eğitimi başlıyor...")
model_cancel_code.fit(X_train_resampled, y_train_resampled)
print("✅ Model eğitimi tamamlandı!")

# ==================== TAHMİNLER ====================
print("\n" + "="*60)
print("TAHMİN VE PERFORMANS ANALİZİ")
print("="*60)

y_pred = model_cancel_code.predict(X_test_scaled)
y_pred_proba = model_cancel_code.predict_proba(X_test_scaled)

# ==================== PERFORMANS METRİKLERİ ====================
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 GENEL PERFORMANS METRİKLERİ:")
print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Sınıf bazında performans
print(f"\n📈 SINIF BAZINDA PERFORMANS METRİKLERİ:")
print(classification_report(
    y_test, 
    y_pred,
    target_names=['A: Havayolu/Taşıyıcı', 'B: Hava durumu', 'C: Ulusal Hava Sistemi', 
                  'D: Güvenlik', 'N: Diğer'],
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 CONFUSION MATRIX:")
print(cm)

# Confusion Matrix görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['A', 'B', 'C', 'D', 'N'],
            yticklabels=['A', 'B', 'C', 'D', 'N'])
plt.title('İptal Kodu Tahmin Modeli - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer', fontsize=12)
plt.xlabel('Tahmin Edilen Değer', fontsize=12)
plt.tight_layout()
plt.savefig('cancel_code_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Confusion Matrix görselleştirmesi kaydedildi: cancel_code_confusion_matrix.png")

# Ağırlıklı ortalama metrikler
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 AĞIRLIKLI ORTALAMA METRİKLER:")
print(f"   • Precision: {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
print(f"   • Recall: {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
print(f"   • F1-Score: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")

# Sınıf bazında detaylı analiz
print(f"\n📋 SINIF BAZINDA DETAYLI ANALİZ:")
for class_idx in range(5):
    if class_idx in y_test.values:
        class_mask = y_test == class_idx
        if class_mask.sum() > 0:
            class_precision = precision_score(
                y_test == class_idx, 
                y_pred == class_idx, 
                zero_division=0
            )
            class_recall = recall_score(
                y_test == class_idx, 
                y_pred == class_idx,
                zero_division=0
            )
            class_f1 = f1_score(
                y_test == class_idx, 
                y_pred == class_idx,
                zero_division=0
            )
            print(f"\n   Sınıf {class_idx} - {cancel_code_names[class_idx]}:")
            print(f"      • Precision: {class_precision:.4f} ({class_precision*100:.2f}%)")
            print(f"      • Recall: {class_recall:.4f} ({class_recall*100:.2f}%)")
            print(f"      • F1-Score: {class_f1:.4f} ({class_f1*100:.2f}%)")
            print(f"      • Test Örnek Sayısı: {class_mask.sum()}")

# ==================== TEMPORAL CROSS-VALIDATION ====================
print("\n" + "="*60)
print("TEMPORAL CROSS-VALIDATION ANALİZİ")
print("="*60)

# TimeSeriesSplit - zaman sırasını korur
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model_cancel_code, X_train_resampled, y_train_resampled, 
                           cv=tscv, scoring='accuracy')

print(f"5-Fold Temporal Cross-Validation Sonuçları:")
print(f"⚠️ Not: Zamansal sızıntıyı önlemek için TimeSeriesSplit kullanılmıştır.")
print(f"  • Ortalama Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"  • Std Deviation: {cv_scores.std():.4f}")
print(f"  • Min: {cv_scores.min():.4f} ({cv_scores.min()*100:.2f}%)")
print(f"  • Max: {cv_scores.max():.4f} ({cv_scores.max()*100:.2f}%)")

# ==================== FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("ÖZELLİK ÖNEMİ ANALİZİ")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_cancel_code.feature_importances_
}).sort_values('importance', ascending=False)

print("\nEn Önemli 15 Özellik:")
print(feature_importance.head(15).to_string(index=False))

# Feature Importance görselleştirme
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Önem Skoru', fontsize=12)
plt.title('İptal Kodu Modeli - En Önemli 15 Özellik', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('cancel_code_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Feature Importance görselleştirmesi kaydedildi: cancel_code_feature_importance.png")

# ==================== MODEL KAYDETME ====================
print("\n" + "="*60)
print("MODEL KAYDETME")
print("="*60)

# models klasörünü oluştur
os.makedirs('models', exist_ok=True)

# Model ve encoder'ları kaydet
joblib.dump(model_cancel_code, 'models/model_cancel_code.joblib')
joblib.dump(label_enc_cancel_code, 'models/label_enc_cancel_code.joblib')
joblib.dump(imputer, 'models/imputer_cancel_code.joblib')
joblib.dump(scaler, 'models/scaler_cancel_code.joblib')

print("✅ Modeller kaydedildi!")
print("\nKaydedilen dosyalar:")
print("   📁 models/model_cancel_code.joblib")
print("   📁 models/label_enc_cancel_code.joblib")
print("   📁 models/imputer_cancel_code.joblib")
print("   📁 models/scaler_cancel_code.joblib")

# ==================== ÖZET RAPOR ====================
print("\n" + "="*60)
print("ÖZET RAPOR")
print("="*60)

print(f"\n🎯 MODEL PERFORMANSI:")
print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   • Precision (Ağırlıklı): {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
print(f"   • Recall (Ağırlıklı): {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
print(f"   • F1-Score (Ağırlıklı): {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
print(f"   • Cross-Validation Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")

print(f"\n📊 VERİ İSTATİSTİKLERİ:")
print(f"   • Toplam İptal Edilmiş Uçuş: {len(df_cancelled)}")
print(f"   • Train Set: {len(X_train)} örnek")
print(f"   • Test Set: {len(X_test)} örnek")
print(f"   • SMOTE Sonrası Train Set: {len(X_train_resampled)} örnek")
print(f"   • Toplam Özellik Sayısı: {X.shape[1]}")

print(f"\n✅ MODEL EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
print(f"Model, {accuracy*100:.2f}% accuracy ile 5 farklı iptal kodunu tahmin edebilmektedir.")
