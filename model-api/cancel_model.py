# LEGACY (kept for the record): superseded by model-api/revision/. This version contains known problems that the revision removes
# (target-based features computed on all data, SMOTE before temporal CV, test-set threshold selection, delay classes built from standardised minutes).
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
from google.colab import drive
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Google Drive'ı bağlama
drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Bitirme')

# Veri setini okuma
df = pd.read_csv("flights_Cancel_final.csv")

# [Önceki veri ön işleme kodları aynı...]
# Tarih işleme - YEAR'ın etkisini azaltmak için normalizasyon
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], format='%Y-%m-%d')
min_year = df['FL_DATE'].dt.year.min()
max_year = df['FL_DATE'].dt.year.max()

df['YEAR_NORMALIZED'] = (df['FL_DATE'].dt.year - min_year) / (max_year - min_year)
df['MONTH'] = df['FL_DATE'].dt.month
df['DAY'] = df['FL_DATE'].dt.day
df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek

# Mevsimsel özellikler - daha detaylı
df['SEASON'] = df['MONTH'].map({12: 0, 1: 0, 2: 0,  # Kış
                                3: 1, 4: 1, 5: 1,   # İlkbahar
                                6: 2, 7: 2, 8: 2,   # Yaz
                                9: 3, 10: 3, 11: 3}) # Sonbahar

# Trigonometric features for cyclical data
df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY'] / 31)
df['DAY_COS'] = np.cos(2 * np.pi * df['DAY'] / 31)

# Hava durumu composite score - daha sofistike
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

# Uçuş zamanı kategorisi - daha detaylı
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

# Mesafe kategorisi - daha dengeli
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

print("Class Dağılımı:")
print(df['CANCELLED'].value_counts())
print(f"İptal oranı: {df['CANCELLED'].mean():.3f}")

# Özellikleri ve hedefleri hazırlama
y_cancelled = df['CANCELLED']
y_cancel_code = df['CANCELLATION_CODE_ENCODED']
X = df.drop(["CANCELLED", "CANCELLATION_CODE_ENCODED", "DEP_TIME"], axis=1)

print(f"Toplam özellik sayısı: {X.shape[1]}")

# ==================== TEMPORAL VALIDATION: Yıllara Göre Bölme ====================
# Zamansal sızıntıyı (temporal leakage) önlemek için veriyi yıllara göre bölelim
# Eğitim: 2016-2022, Test: 2023-2024
print("\n" + "="*60)
print("TEMPORAL VALIDATION - ZAMANSAL DOĞRULAMA")
print("="*60)

# YEAR_NORMALIZED'dan gerçek yılı geri hesapla
df['YEAR'] = (df['YEAR_NORMALIZED'] * (max_year - min_year)) + min_year
df['YEAR'] = df['YEAR'].astype(int)

train_mask = df['YEAR'] <= 2022
test_mask = df['YEAR'] >= 2023

X_train = X[train_mask]
X_test = X[test_mask]
y_cancelled_train = y_cancelled[train_mask]
y_cancelled_test = y_cancelled[test_mask]
y_cancel_code_train = y_cancel_code[train_mask]
y_cancel_code_test = y_cancel_code[test_mask]

print(f"Eğitim verisi: {len(X_train)} örnek (2016-2022)")
print(f"Test verisi: {len(X_test)} örnek (2023-2024)")
print(f"Eğitim yılları: {sorted(df[train_mask]['YEAR'].unique())}")
print(f"Test yılları: {sorted(df[test_mask]['YEAR'].unique())}")
print(f"Eğitim iptal oranı: {y_cancelled_train.mean():.3f}")
print(f"Test iptal oranı: {y_cancelled_test.mean():.3f}")

# Eksik değerleri doldurma
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# ==================== GELİŞMİŞ MODEL EĞİTİMİ ====================
print("\n" + "="*50)
print("GELİŞMİŞ İPTAL MODELİ EĞİTİMİ")
print("="*50)

# SMOTE
smote = SMOTE(sampling_strategy=0.15, random_state=42, k_neighbors=3)
X_train_resampled, y_cancelled_train_resampled = smote.fit_resample(X_train_scaled, y_cancelled_train)

# İptal modeli
model_cancelled = XGBClassifier(
    scale_pos_weight=4.0,
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2.0,
    reg_lambda=2.0,
    min_child_weight=20,
    gamma=2.0,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# Model eğitimi
model_cancelled.fit(X_train_resampled, y_cancelled_train_resampled)

# Tahminler
y_pred_cancelled_proba = model_cancelled.predict_proba(X_test_scaled)
best_threshold = 0.5
y_pred_cancelled_final = (y_pred_cancelled_proba[:, 1] >= best_threshold).astype(int)

# ==================== KAPSAMLI PERFORMANS ANALİZİ ====================
print("\n" + "="*60)
print("KAPSAMLI PERFORMANS ANALİZİ")
print("="*60)

def comprehensive_performance_analysis(y_true, y_pred, y_pred_proba, model_name="Model"):
    """Kapsamlı performans analizi fonksiyonu"""
    
    print(f"\n{'-'*40}")
    print(f"{model_name} - DETAYLI PERFORMANS METRİKLERİ")
    print(f"{'-'*40}")
    
    # Temel metrikler
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"📊 Temel Metrikler:")
    print(f"   • Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"   • Kesinlik (Precision): {precision:.4f}")
    print(f"   • Duyarlılık (Recall): {recall:.4f}")
    print(f"   • F1 Skoru: {f1:.4f}")
    print(f"   • ROC-AUC Skoru: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📈 Confusion Matrix Analizi:")
    print(f"   • True Negatives (TN): {tn}")
    print(f"   • False Positives (FP): {fp}")
    print(f"   • False Negatives (FN): {fn}")
    print(f"   • True Positives (TP): {tp}")
    
    # Hata oranları
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    
    print(f"\n⚠️ Hata Analizi:")
    print(f"   • Yanlış Pozitif Oranı (FPR): {fpr:.4f}")
    print(f"   • Yanlış Negatif Oranı (FNR): {fnr:.4f}")
    
    # İş dünyası etkileri
    print(f"\n💼 İŞ DÜNYASI ETKİLERİ:")
    
    # False Positives (Gereksiz iptal tahminleri)
    print(f"   🔴 YANLIŞ POZİTİF ETKILER (FP={fp}):")
    print(f"      - Gereksiz yolcu bildirimleri")
    print(f"      - Müşteri memnuniyetsizliği")
    print(f"      - Alternatif uçuş aramaları")
    print(f"      - Operasyonel karmaşa")
    
    # False Negatives (Kaçırılan iptal tahminleri)
    print(f"   🔴 YANLIŞ NEGATİF ETKILER (FN={fn}):")
    print(f"      - Hazırlıksız iptal durumları")
    print(f"      - Yolcu mağduriyeti")
    print(f"      - Havaalanında bekleme")
    print(f"      - Tazminat maliyetleri")
    
    # Risk değerlendirmesi
    if fpr > 0.1:
        print(f"   ⚠️ YÜKSEK FPR RİSKİ: Çok fazla gereksiz iptal tahmini!")
    if fnr > 0.2:
        print(f"   ⚠️ YÜKSEK FNR RİSKİ: Çok fazla iptal kaçırılıyor!")
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'roc_auc': roc_auc, 'fpr': fpr, 'fnr': fnr,
        'confusion_matrix': cm
    }

# Performans analizi çalıştır
performance_results = comprehensive_performance_analysis(
    y_cancelled_test, y_pred_cancelled_final, y_pred_cancelled_proba[:, 1], 
    "Uçuş İptal Tahmin Modeli"
)

# ==================== SCENARIO-BASED PERFORMANCE ANALYSIS ====================
print("\n" + "="*60)
print("SENARYOYA DAYALI PERFORMANS ANALİZİ")
print("="*60)

def scenario_performance_analysis(X_test, y_test, y_pred, y_pred_proba, feature_names):
    """Farklı senaryolarda model performansını analiz eder"""
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['actual'] = y_test.values
    test_df['predicted'] = y_pred
    test_df['pred_proba'] = y_pred_proba
    
    scenarios = {
        'Kötü Hava': test_df['WEATHER_COMPOSITE'] > test_df['WEATHER_COMPOSITE'].quantile(0.8),
        'İyi Hava': test_df['WEATHER_COMPOSITE'] < test_df['WEATHER_COMPOSITE'].quantile(0.2),
        'Kış Ayları': test_df['SEASON'] == 0,
        'Yaz Ayları': test_df['SEASON'] == 2,
        'Uzun Mesafe': test_df['DISTANCE_CATEGORY'] >= 3,
        'Kısa Mesafe': test_df['DISTANCE_CATEGORY'] <= 1,
        'Yoğun Saatler': test_df['DEP_TIME_DETAILED'].isin([1, 4]),
        'Sakin Saatler': test_df['DEP_TIME_DETAILED'].isin([3, 6]),
        'Güvenilir Havayolu': test_df['AIRLINE_RELIABILITY'] < test_df['AIRLINE_RELIABILITY'].quantile(0.3),
        'Sorunlu Havayolu': test_df['AIRLINE_RELIABILITY'] > test_df['AIRLINE_RELIABILITY'].quantile(0.7)
    }
    
    print(f"{'Senaryo':<20} {'Veri':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Güçlü/Zayıf'}")
    print("-" * 80)
    
    scenario_results = {}
    
    for scenario_name, mask in scenarios.items():
        if mask.sum() > 10:  # En az 10 örnek olmalı
            scenario_actual = test_df.loc[mask, 'actual']
            scenario_pred = test_df.loc[mask, 'predicted']
            scenario_proba = test_df.loc[mask, 'pred_proba']
            
            if len(scenario_actual.unique()) > 1:  # Both classes present
                acc = accuracy_score(scenario_actual, scenario_pred)
                prec = precision_score(scenario_actual, scenario_pred, zero_division=0)
                rec = recall_score(scenario_actual, scenario_pred, zero_division=0)
                f1 = f1_score(scenario_actual, scenario_pred, zero_division=0)
                
                # Performans değerlendirmesi
                performance_score = (acc + prec + rec + f1) / 4
                strength = "🟢 Güçlü" if performance_score > 0.7 else "🔴 Zayıf" if performance_score < 0.5 else "🟡 Orta"
                
                print(f"{scenario_name:<20} {mask.sum():<8} {acc:<10.3f} {prec:<10.3f} {rec:<8.3f} {f1:<8.3f} {strength}")
                
                scenario_results[scenario_name] = {
                    'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                    'sample_size': mask.sum(), 'performance_score': performance_score
                }
    
    return scenario_results

# Senaryo analizi çalıştır
scenario_results = scenario_performance_analysis(
    X_test_scaled, y_cancelled_test, y_pred_cancelled_final, 
    y_pred_cancelled_proba[:, 1], X.columns
)

# ==================== EXPLAINABLE AI (XAI) ANALİZİ ====================
print("\n" + "="*60)
print("EXPLAINABLE AI (XAI) ANALİZİ")
print("="*60)

# Feature Importance Analizi
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_cancelled.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 EN ÖNEMLİ ÖZELLİKLER (Top 15):")
print("-" * 50)

feature_descriptions = {
    'WEATHER_COMPOSITE': 'Hava Durumu Kompozit Skoru',
    'AIRLINE_RELIABILITY': 'Havayolu Güvenilirlik Skoru',
    'DISTANCE': 'Uçuş Mesafesi',
    'DISTANCE_CATEGORY': 'Mesafe Kategorisi',
    'wspd': 'Rüzgar Hızı',
    'prcp': 'Yağış Miktarı',
    'snow': 'Kar Miktarı',
    'tmin': 'Minimum Sıcaklık',
    'tmax': 'Maksimum Sıcaklık',
    'pres': 'Atmosfer Basıncı',
    'MONTH': 'Ay',
    'SEASON': 'Mevsim',
    'DEP_TIME_DETAILED': 'Detaylı Kalkış Zamanı',
    'DAY_OF_WEEK': 'Haftanın Günü',
    'ROUTE_POPULARITY_LOG': 'Rota Popülaritesi',
    'AIR': 'Havayolu Kodu',
    'ORG': 'Çıkış Havaalanı',
    'DST': 'Varış Havaalanı'
}

for i, (idx, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    feature_name = row['feature']
    importance = row['importance']
    description = feature_descriptions.get(feature_name, feature_name)
    
    # Önem seviyesi belirleme
    if importance > 0.1:
        level = "🔴 Kritik"
    elif importance > 0.05:
        level = "🟡 Önemli"
    else:
        level = "🟢 Yardımcı"
    
    print(f"{i:2d}. {feature_name:<25} | {importance:6.4f} | {level} | {description}")

# ==================== FEATURE INTERACTION ANALİZİ ====================
print("\n" + "="*60)
print("ÖZELLİK ETKİLEŞİM ANALİZİ")
print("="*60)

def analyze_feature_interactions(df, target_col='CANCELLED', top_n=10):
    """Özelliklerin hedef değişkenle ilişkilerini analiz eder"""
    
    print("\n📊 ÖZELLİKLERİN İPTAL ORANI ÜZERİNDEKİ ETKİSİ:")
    print("-" * 60)
    
    interactions = {}
    
    # Hava durumu etkisi
    weather_bins = pd.cut(df['WEATHER_COMPOSITE'], bins=5, labels=['Çok İyi', 'İyi', 'Orta', 'Kötü', 'Çok Kötü'])
    weather_cancel_rate = df.groupby(weather_bins)[target_col].mean()
    
    print("🌤️ HAVA DURUMU ETKİSİ:")
    for weather, rate in weather_cancel_rate.items():
        print(f"   {weather}: {rate:.3f} ({rate*100:.1f}%)")
    
    # Mevsimsel etki
    season_names = {0: 'Kış', 1: 'İlkbahar', 2: 'Yaz', 3: 'Sonbahar'}
    season_cancel_rate = df.groupby('SEASON')[target_col].mean()
    
    print("\n🗓️ MEVSIMSEL ETKİ:")
    for season, rate in season_cancel_rate.items():
        season_name = season_names.get(season, f'Sezon {season}')
        print(f"   {season_name}: {rate:.3f} ({rate*100:.1f}%)")
    
    # Uçuş zamanı etkisi
    time_names = {1: 'Erken Sabah', 2: 'Sabah', 3: 'Öğlen', 4: 'Öğleden Sonra', 5: 'Akşam', 6: 'Gece'}
    time_cancel_rate = df.groupby('DEP_TIME_DETAILED')[target_col].mean()
    
    print("\n🕐 UÇUŞ ZAMANI ETKİSİ:")
    for time_code, rate in time_cancel_rate.items():
        time_name = time_names.get(time_code, f'Zaman {time_code}')
        print(f"   {time_name}: {rate:.3f} ({rate*100:.1f}%)")
    
    # Mesafe etkisi
    distance_cancel_rate = df.groupby('DISTANCE_CATEGORY')[target_col].mean()
    distance_names = {0: 'Çok Kısa', 1: 'Kısa', 2: 'Orta', 3: 'Uzun', 4: 'Çok Uzun'}
    
    print("\n✈️ MESAFE ETKİSİ:")
    for dist_code, rate in distance_cancel_rate.items():
        if not pd.isna(dist_code):
            dist_name = distance_names.get(dist_code, f'Mesafe {dist_code}')
            print(f"   {dist_name}: {rate:.3f} ({rate*100:.1f}%)")
    
    return interactions

# Etkileşim analizi çalıştır
feature_interactions = analyze_feature_interactions(df)

# ==================== THRESHOLD OPTİMİZASYONU ====================
print("\n" + "="*60)
print("GELISMIS THRESHOLD OPTİMİZASYONU")
print("="*60)

def optimize_threshold_advanced(y_true, y_pred_proba):
    """Gelişmiş threshold optimizasyonu - iş etkilerini dikkate alarak"""
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    # İş maliyetleri (varsayılan değerler)
    COST_FP = 50    # Gereksiz iptal tahmini maliyeti (yolcu memnuniyetsizliği vs.)
    COST_FN = 200   # Kaçırılan iptal maliyeti (tazminat, müşteri kaybı vs.)
    
    print(f"📊 THRESHOLD OPTİMİZASYONU (İş Maliyetleri Dahil):")
    print(f"   💰 False Positive Maliyeti: {COST_FP}$ per case")
    print(f"   💰 False Negative Maliyeti: {COST_FN}$ per case")
    print()
    print(f"{'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Total Cost':<12} {'Öneri'}")
    print("-" * 80)
    
    best_cost = float('inf')
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Metrikleri hesapla
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Toplam maliyet
        total_cost = (fp * COST_FP) + (fn * COST_FN)
        
        # En iyi threshold'u bul
        recommendation = ""
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold
            recommendation = "🏆 EN İYİ"
        elif total_cost < best_cost * 1.05:  # %5 tolerans
            recommendation = "⭐ İYİ"
        
        print(f"{threshold:<12.2f} {accuracy:<10.3f} {precision:<10.3f} {recall:<8.3f} {f1:<8.3f} {total_cost:<12.0f} {recommendation}")
        
        results.append({
            'threshold': threshold, 'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1, 'total_cost': total_cost, 'fp': fp, 'fn': fn
        })
    
    print(f"\n🎯 ÖNERİLEN THRESHOLD: {best_threshold:.2f}")
    print(f"💰 MİNİMUM MALİYET: {best_cost:.0f}$")
    
    return best_threshold, results

# Threshold optimizasyonu
optimal_threshold, threshold_results = optimize_threshold_advanced(
    y_cancelled_test, y_pred_cancelled_proba[:, 1]
)

# ==================== MODEL GÜVENİLİRLİK ANALİZİ ====================
print("\n" + "="*60)
print("MODEL GÜVENİLİRLİK ANALİZİ")
print("="*60)

def model_reliability_analysis(model, X_test, y_test, n_bootstrap=100):
    """Model güvenilirlik analizi - bootstrap ile"""
    
    print("🔄 Bootstrap Analizi Başlatılıyor...")
    
    n_samples = len(X_test)
    bootstrap_scores = []
    
    for i in range(n_bootstrap):
        if i % 20 == 0:
            print(f"   Bootstrap iterasyonu: {i}/{n_bootstrap}")
        
        # Bootstrap sampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_test[indices]
        y_boot = y_test.iloc[indices]
        
        # Tahmin ve skor
        y_pred_boot = model.predict(X_boot)
        score = accuracy_score(y_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # İstatistikler
    mean_score = np.mean(bootstrap_scores)
    std_score = np.std(bootstrap_scores)
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    
    print(f"\n📈 MODEL GÜVENİLİRLİK SONUÇLARI:")
    print(f"   • Ortalama Doğruluk: {mean_score:.4f}")
    print(f"   • Standart Sapma: {std_score:.4f}")
    print(f"   • %95 Güven Aralığı: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Güvenilirlik değerlendirmesi
    reliability_score = 1 - (std_score / mean_score)  # Coefficient of variation
    
    if reliability_score > 0.95:
        reliability_level = "🟢 Çok Güvenilir"
    elif reliability_score > 0.90:
        reliability_level = "🟡 Güvenilir"
    else:
        reliability_level = "🔴 Güvenilirlik Sorunu"
    
    print(f"   • Güvenilirlik Skoru: {reliability_score:.4f}")
    print(f"   • Güvenilirlik Seviyesi: {reliability_level}")
    
    return {
        'mean_score': mean_score, 'std_score': std_score,
        'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'reliability_score': reliability_score
    }

# Model güvenilirlik analizi
reliability_results = model_reliability_analysis(model_cancelled, X_test_scaled, y_cancelled_test)

# ==================== TEMPORAL CROSS-VALIDATION ANALİZİ ====================
print("\n" + "="*60)
print("TEMPORAL CROSS-VALIDATION STABİLİTE ANALİZİ")
print("="*60)

def temporal_cross_validation_analysis(model, X, y, cv_folds=5):
    """TimeSeriesSplit ile zamansal cross-validation analizi"""
    
    print(f"🔄 {cv_folds}-Fold Temporal Cross-Validation Analizi...")
    print("⚠️ Not: Zamansal sızıntıyı önlemek için TimeSeriesSplit kullanılmıştır.")
    
    # TimeSeriesSplit - zaman sırasını korur
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Farklı metrikler için CV skorları
    accuracy_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=tscv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=tscv, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=tscv, scoring='f1')
    roc_auc_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
    
    cv_results = {
        'Accuracy': accuracy_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1': f1_scores,
        'ROC-AUC': roc_auc_scores
    }
    
    print(f"\n📊 TEMPORAL CROSS-VALIDATION SONUÇLARI:")
    print(f"{'Metrik':<12} {'Ortalama':<10} {'Std':<8} {'Min':<8} {'Max':<8} {'Stabilite'}")
    print("-" * 65)
    
    for metric_name, scores in cv_results.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Stabilite değerlendirmesi
        cv_stability = 1 - (std_score / mean_score) if mean_score > 0 else 0
        if cv_stability > 0.95:
            stability_level = "🟢 Stabil"
        elif cv_stability > 0.90:
            stability_level = "🟡 Orta"
        else:
            stability_level = "🔴 Değişken"
        
        print(f"{metric_name:<12} {mean_score:<10.4f} {std_score:<8.4f} {min_score:<8.4f} {max_score:<8.4f} {stability_level}")
    
    return cv_results

# Temporal cross-validation analizi (sadece eğitim verisi üzerinde)
cv_results = temporal_cross_validation_analysis(model_cancelled, X_train_resampled, y_cancelled_train_resampled)

# ==================== HATA ANALİZİ VE İYİLEŞTİRME ÖNERİLERİ ====================
print("\n" + "="*60)
print("DETAYLI HATA ANALİZİ VE İYİLEŞTİRME ÖNERİLERİ")
print("="*60)

def detailed_error_analysis(X_test, y_test, y_pred, y_pred_proba, feature_names):
    """Detaylı hata analizi ve iyileştirme önerileri"""
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['actual'] = y_test.values
    test_df['predicted'] = y_pred
    test_df['pred_proba'] = y_pred_proba
    
    # Hata türlerini ayır
    false_positives = test_df[(test_df['actual'] == 0) & (test_df['predicted'] == 1)]
    false_negatives = test_df[(test_df['actual'] == 1) & (test_df['predicted'] == 0)]
    
    print(f"🔍 HATA DAĞILIMI:")
    print(f"   • False Positives: {len(false_positives)} (Gereksiz iptal tahminleri)")
    print(f"   • False Negatives: {len(false_negatives)} (Kaçırılan iptaller)")
    
    # False Positives analizi
    if len(false_positives) > 0:
        print(f"\n🔴 FALSE POSITIVE ANALİZİ:")
        print(f"   Ortalama Tahmin Olasılığı: {false_positives['pred_proba'].mean():.3f}")
        
        # En problemli özellikler
        fp_means = false_positives[feature_names].mean()
        overall_means = test_df[feature_names].mean()
        
        print(f"   En Problemli Özellikler:")
        for feature in feature_names[:10]:  # Top 10 feature
            if abs(fp_means[feature] - overall_means[feature]) > 0.1:
                direction = "↑" if fp_means[feature] > overall_means[feature] else "↓"
                print(f"     • {feature}: {direction} {abs(fp_means[feature] - overall_means[feature]):.3f}")
    
    # False Negatives analizi
    if len(false_negatives) > 0:
        print(f"\n🔴 FALSE NEGATIVE ANALİZİ:")
        print(f"   Ortalama Tahmin Olasılığı: {false_negatives['pred_proba'].mean():.3f}")
        
        # En problemli özellikler
        fn_means = false_negatives[feature_names].mean()
        
        print(f"   En Problemli Özellikler:")
        for feature in feature_names[:10]:
            if abs(fn_means[feature] - overall_means[feature]) > 0.1:
                direction = "↑" if fn_means[feature] > overall_means[feature] else "↓"
                print(f"     • {feature}: {direction} {abs(fn_means[feature] - overall_means[feature]):.3f}")
    
    # İyileştirme önerileri
    print(f"\n💡 İYİLEŞTİRME ÖNERİLERİ:")
    
    # False Positive azaltma
    if len(false_positives) > 50:
        print(f"   🎯 FALSE POSITIVE AZALTMA STRATEJİLERİ:")
        print(f"     • Threshold değerini artır (mevcut: {best_threshold:.2f})")
        print(f"     • Precision'ı artırmak için model parametrelerini ayarla")
        print(f"     • Daha fazla negatif örnek ekle")
        print(f"     • Feature engineering ile ayırt edici özellikler ekle")
    
    # False Negative azaltma
    if len(false_negatives) > 20:
        print(f"   🎯 FALSE NEGATIVE AZALTMA STRATEJİLERİ:")
        print(f"     • Threshold değerini azalt")
        print(f"     • SMOTE ile daha fazla pozitif örnek oluştur")
        print(f"     • Recall'ı artırmak için scale_pos_weight artır")
        print(f"     • Ensemble metodları kullan")
    
    # Genel öneriler
    print(f"   🎯 GENEL İYİLEŞTİRME STRATEJİLERİ:")
    print(f"     • Daha fazla hava durumu verisi ekle")
    print(f"     • Havaalanı yoğunluk bilgilerini dahil et")
    print(f"     • Geçmiş iptal geçmişi özelliklerini ekle")
    print(f"     • Seasonal decomposition uygula")
    print(f"     • Deep learning modelleri dene")
    
    return false_positives, false_negatives

# Hata analizi
fp_analysis, fn_analysis = detailed_error_analysis(
    X_test_scaled, y_cancelled_test, y_pred_cancelled_final, 
    y_pred_cancelled_proba[:, 1], X.columns
)

# ==================== BUSINESS IMPACT ANALYSIS ====================
print("\n" + "="*60)
print("İŞ ETKİSİ ANALİZİ VE ROI HESAPLAMA")
print("="*60)

def business_impact_analysis(confusion_matrix, test_size):
    """İş etkisi analizi ve ROI hesaplama"""
    
    tn, fp, fn, tp = confusion_matrix.ravel()
    
    # Varsayılan maliyetler (günlük)
    COST_FP_DAILY = 75      # Gereksiz iptal tahmini maliyeti
    COST_FN_DAILY = 300     # Kaçırılan iptal maliyeti
    REVENUE_TP_DAILY = 150  # Doğru iptal tahmini faydası
    REVENUE_TN_DAILY = 10   # Doğru normal tahmin faydası
    
    # Günlük maliyetler
    daily_fp_cost = fp * COST_FP_DAILY
    daily_fn_cost = fn * COST_FN_DAILY
    daily_tp_revenue = tp * REVENUE_TP_DAILY
    daily_tn_revenue = tn * REVENUE_TN_DAILY
    
    # Net günlük etki
    net_daily_impact = daily_tp_revenue + daily_tn_revenue - daily_fp_cost - daily_fn_cost
    
    # Yıllık projeksiyon
    annual_impact = net_daily_impact * 365
    
    print(f"💰 GÜNLÜK İŞ ETKİSİ ANALİZİ:")
    print(f"   • False Positive Maliyeti: ${daily_fp_cost:,.0f}")
    print(f"   • False Negative Maliyeti: ${daily_fn_cost:,.0f}")
    print(f"   • True Positive Faydası: ${daily_tp_revenue:,.0f}")
    print(f"   • True Negative Faydası: ${daily_tn_revenue:,.0f}")
    print(f"   • Net Günlük Etki: ${net_daily_impact:,.0f}")
    
    print(f"\n📅 YILLIK PROJEKSİYON:")
    print(f"   • Yıllık Net Etki: ${annual_impact:,.0f}")
    
    # ROI hesaplama (varsayılan sistem geliştirme maliyeti)
    SYSTEM_DEVELOPMENT_COST = 50000  # Sistem geliştirme maliyeti
    
    if annual_impact > 0:
        roi = (annual_impact / SYSTEM_DEVELOPMENT_COST) * 100
        payback_period = SYSTEM_DEVELOPMENT_COST / (annual_impact / 365)
        
        print(f"\n📊 ROI ANALİZİ:")
        print(f"   • Sistem Geliştirme Maliyeti: ${SYSTEM_DEVELOPMENT_COST:,.0f}")
        print(f"   • Yıllık ROI: {roi:.1f}%")
        print(f"   • Geri Ödeme Süresi: {payback_period:.0f} gün")
        
        if roi > 100:
            print(f"   • 🟢 Yüksek ROI - Proje Karlı!")
        elif roi > 50:
            print(f"   • 🟡 Orta ROI - Proje Kabul Edilebilir")
        else:
            print(f"   • 🔴 Düşük ROI - Proje Gözden Geçirilmeli")
    else:
        print(f"   • 🔴 Negatif Etki - Model İyileştirme Gerekli")
    
    # Sektör karşılaştırması
    print(f"\n🏭 SEKTÖR KARŞILAŞTIRMASI:")
    print(f"   • Havayolu sektörü iptal maliyeti: $100-500 per flight")
    print(f"   • Modelimizin birim maliyeti: ${(daily_fp_cost + daily_fn_cost) / (fp + fn):.0f}")
    print(f"   • Sektör ortalamasına göre: {'🟢 Düşük' if (daily_fp_cost + daily_fn_cost) / (fp + fn) < 200 else '🔴 Yüksek'}")
    
    return {
        'daily_impact': net_daily_impact,
        'annual_impact': annual_impact,
        'roi': roi if annual_impact > 0 else 0,
        'payback_days': payback_period if annual_impact > 0 else float('inf')
    }

# İş etkisi analizi
business_results = business_impact_analysis(performance_results['confusion_matrix'], len(y_cancelled_test))

# ==================== VİZUALİZASYON VE RAPORLAMA ====================
print("\n" + "="*60)
print("GÖRSELLEŞTİRME VE RAPORLAMA")
print("="*60)

# Çoklu grafik oluştur
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Uçuş İptal Tahmin Modeli - Kapsamlı Analiz', fontsize=16, fontweight='bold')

# 1. Feature Importance
axes[0, 0].barh(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
axes[0, 0].set_title('En Önemli 10 Özellik')
axes[0, 0].set_xlabel('Önem Skoru')

# 2. Confusion Matrix
cm = performance_results['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_xlabel('Tahmin')
axes[0, 1].set_ylabel('Gerçek')

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_cancelled_test, y_pred_cancelled_proba[:, 1])
axes[0, 2].plot(fpr, tpr, label=f'ROC Curve (AUC = {performance_results["roc_auc"]:.3f})')
axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].set_title('ROC Curve')
axes[0, 2].legend()

# 4. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_cancelled_test, y_pred_cancelled_proba[:, 1])
axes[1, 0].plot(recall, precision, label=f'PR Curve (AUC = {auc(recall, precision):.3f})')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curve')
axes[1, 0].legend()

# 5. Threshold Analysis
threshold_df = pd.DataFrame(threshold_results)
axes[1, 1].plot(threshold_df['threshold'], threshold_df['f1'], label='F1 Score', marker='o')
axes[1, 1].plot(threshold_df['threshold'], threshold_df['precision'], label='Precision', marker='s')
axes[1, 1].plot(threshold_df['threshold'], threshold_df['recall'], label='Recall', marker='^')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Threshold vs Metrikler')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Scenario Performance
scenario_names = list(scenario_results.keys())[:8]  # İlk 8 senaryo
scenario_f1s = [scenario_results[name]['f1'] for name in scenario_names]
axes[1, 2].bar(range(len(scenario_names)), scenario_f1s)
axes[1, 2].set_xticks(range(len(scenario_names)))
axes[1, 2].set_xticklabels(scenario_names, rotation=45, ha='right')
axes[1, 2].set_ylabel('F1 Score')
axes[1, 2].set_title('Senaryolara Göre Performans')

plt.tight_layout()
plt.show()

# ==================== ÖZET RAPOR ====================
print("\n" + "="*60)
print("EXECUTIVE SUMMARY - YÖNETİCİ ÖZETİ")
print("="*60)

def generate_executive_summary(performance_results, business_results, reliability_results):
    """Yönetici özeti oluştur"""
    
    print("📋 YÖNETİCİ ÖZETİ RAPORU")
    print("=" * 50)
    
    # Genel performans
    print(f"🎯 MODEL PERFORMANSI:")
    print(f"   • Genel Doğruluk: {performance_results['accuracy']:.1%}")
    print(f"   • Kesinlik (Precision): {performance_results['precision']:.1%}")
    print(f"   • Duyarlılık (Recall): {performance_results['recall']:.1%}")
    print(f"   • F1 Skoru: {performance_results['f1']:.3f}")
    print(f"   • ROC-AUC: {performance_results['roc_auc']:.3f}")
    
    # Güvenilirlik
    print(f"\n🔒 MODEL GÜVENİLİRLİĞİ:")
    print(f"   • Güvenilirlik Skoru: {reliability_results['reliability_score']:.1%}")
    print(f"   • %95 Güven Aralığı: [{reliability_results['ci_lower']:.3f}, {reliability_results['ci_upper']:.3f}]")
    
    # İş etkisi
    print(f"\n💰 İŞ ETKİSİ:")
    print(f"   • Günlük Net Etki: ${business_results['daily_impact']:,.0f}")
    print(f"   • Yıllık Net Etki: ${business_results['annual_impact']:,.0f}")
    print(f"   • ROI: {business_results['roi']:.1f}%")
    print(f"   • Geri Ödeme Süresi: {business_results['payback_days']:.0f} gün")
    
    # Öneriler
    print(f"\n🎯 STRATEJİK ÖNERİLER:")
    if performance_results['f1'] > 0.7:
        print(f"   ✅ Model prodüksiyona hazır")
    else:
        print(f"   ⚠️ Model iyileştirme gerekli")
    
    if business_results['roi'] > 100:
        print(f"   ✅ Yüksek ROI - Yatırım önerilir")
    else:
        print(f"   ⚠️ ROI düşük - Optimizasyon gerekli")
    
    if reliability_results['reliability_score'] > 0.9:
        print(f"   ✅ Model güvenilir")
    else:
        print(f"   ⚠️ Güvenilirlik artırılmalı")
    
    print(f"\n📈 SONRAKI ADIMLAR:")
    print(f"   1. A/B testi ile pilot uygulama")
    print(f"   2. Real-time veri entegrasyonu")
    print(f"   3. Müşteri feedback sistemi kurulumu")
    print(f"   4. Otomatik model güncelleme sistemi")
    print(f"   5. Dashboard ve alerting sistemi")

# Özet rapor oluştur
generate_executive_summary(performance_results, business_results, reliability_results)

# ==================== MODEL KAYDETME (GELİŞMİŞ) ====================
print("\n" + "="*60)
print("GELİŞMİŞ MODEL KAYDETME SİSTEMİ")
print("="*60)

# Ana modeller
joblib.dump(model_cancelled, 'model_cancelled_advanced.joblib')
joblib.dump(imputer, 'imputer_advanced.joblib')
joblib.dump(scaler, 'scaler_advanced.joblib')

# Gelişmiş model konfigürasyonu
advanced_config = {
    'features': list(X.columns),
    'optimal_threshold': optimal_threshold,
    'min_year': min_year,
    'max_year': max_year,
    'performance_metrics': performance_results,
    'business_metrics': business_results,
    'reliability_metrics': reliability_results,
    'feature_importance': feature_importance.to_dict(),
    'scenario_performance': scenario_results,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_version': '2.0_advanced'
}

joblib.dump(advanced_config, 'model_config_advanced.joblib')

# Analiz sonuçları
analysis_results = {
    'threshold_analysis': threshold_results,
    'cv_results': cv_results,
    'feature_interactions': feature_interactions,
    'false_positive_analysis': fp_analysis.to_dict() if len(fp_analysis) > 0 else {},
    'false_negative_analysis': fn_analysis.to_dict() if len(fn_analysis) > 0 else {}
}

joblib.dump(analysis_results, 'analysis_results.joblib')

print("✅ GELİŞMİŞ MODEL KAYDETME TAMAMLANDI")
print("\nKaydedilen dosyalar:")
advanced_files = [
    'model_cancelled_advanced.joblib',
    'imputer_advanced.joblib', 
    'scaler_advanced.joblib',
    'model_config_advanced.joblib',
    'analysis_results.joblib'
]

for file in advanced_files:
    print(f"   📁 {file}")

print(f"\n💡 KULLANIM ÖRNEĞİ:")
print("# Model yükleme")
print("model = joblib.load('model_cancelled_advanced.joblib')")
print("config = joblib.load('model_config_advanced.joblib')")
print("optimal_threshold = config['optimal_threshold']")
print()
print("# Tahmin yapma")
print("prediction_proba = model.predict_proba(new_data)")
print("prediction = (prediction_proba[:, 1] >= optimal_threshold).astype(int)")

print(f"\n🎉 GELIŞMIŞ UÇUŞ İPTAL TAHMİN SİSTEMİ HAZIR!")
print(f"Sistem, {performance_results['accuracy']:.1%} doğruluk oranı ile yılda ${business_results['annual_impact']:,.0f} tasarruf sağlayabilir.")