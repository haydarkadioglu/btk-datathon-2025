# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# %%
# ==============================================================================
# 1. VERİ YÜKLEME VE TEMEL İŞLEME
# ==============================================================================
print("--- Adım 1: Veri Yükleniyor ve Temel Özellikler Oluşturuluyor ---")

try:
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı. Lütfen 'train.csv' ve 'test.csv' dosyalarının doğru yolda olduğundan emin olun. Detay: {e}")
    exit()

df_full = pd.concat([df_train, df_test], sort=False)
df_full['event_time'] = pd.to_datetime(df_full['event_time'])
df_full['day_of_week'] = df_full['event_time'].dt.dayofweek
df_full['is_weekend'] = df_full['day_of_week'].isin([5, 6]).astype(int)

# %%
# ==============================================================================
# 2. GELİŞMİŞ ÖZELLİK MÜHENDİSLİĞİ (PRE-AGGREGATION)
# ==============================================================================
print("--- Adım 2: Gelişmiş Özellik Mühendisliği (Oturum Öncesi) ---")

# --- 2.1 Kullanıcı Geçmişi Özellikleri ---
print("   - Kullanıcı geçmişi özellikleri oluşturuluyor...")
user_features = df_train.groupby('user_id').agg(
    user_avg_session_value=('session_value', 'mean'),
    user_session_count=('user_session', 'nunique'),
    user_buy_count=('event_type', lambda x: (x == 'BUY').sum())
).reset_index()

# --- 2.2 Tahmini Ürün Fiyat Seviyesi ---
print("   - Tahmini ürün fiyat seviyeleri hesaplanıyor...")
product_prices = df_train[df_train['event_type'] == 'BUY'].groupby('product_id')['session_value'].mean().rename('inferred_price')
global_mean_price = product_prices.mean()

# --- 2.3 YENİ ÖZELLİK: Kategori Popülerliği ---
print("   - Kategori popülerliği hesaplanıyor...")
category_popularity = df_train[df_train['event_type'] == 'BUY']['category_id'].value_counts().rename('category_popularity')
global_mean_popularity = category_popularity.mean()

# --- Özellikleri ana veri setine ekleme ---
df_full = pd.merge(df_full, user_features, on='user_id', how='left')
df_full = pd.merge(df_full, product_prices, on='product_id', how='left')
df_full = pd.merge(df_full, category_popularity, on='category_id', how='left')

# NaN Değerlerini Doldurma
df_full['user_avg_session_value'].fillna(df_train['session_value'].mean(), inplace=True)
df_full['user_session_count'].fillna(0, inplace=True)
df_full['user_buy_count'].fillna(0, inplace=True)
df_full['inferred_price'].fillna(global_mean_price, inplace=True)
df_full['category_popularity'].fillna(global_mean_popularity, inplace=True)

# %%
# ==============================================================================
# 3. OTURUM BAZINDA VERİ BİRLEŞTİRME (AGGREGATION)
# ==============================================================================
print("--- Adım 3: Veri Oturum Bazında Birleştiriliyor ---")

df_full = pd.get_dummies(df_full, columns=['event_type'], prefix='event')

session_df = df_full.groupby('user_session').agg(
    # Temel event sayıları
    views=('event_VIEW', 'sum'),
    add_to_carts=('event_ADD_CART', 'sum'),
    removals_from_cart=('event_REMOVE_CART', 'sum'),
    buys=('event_BUY', 'sum'),
    # Zaman ve fiyat özellikleri
    session_duration_seconds=('event_time', lambda x: (x.max() - x.min()).total_seconds()),
    avg_inferred_price=('inferred_price', 'mean'),
    is_weekend=('is_weekend', 'max'),
    # Kullanıcı geçmişi
    user_avg_session_value=('user_avg_session_value', 'first'),
    user_session_count=('user_session_count', 'first'),
    user_buy_count=('user_buy_count', 'first'),
    # YENİ ÖZELLİK: Kategori
    avg_category_popularity=('category_popularity', 'mean'),
    main_category=('category_id', lambda x: x.mode()[0] if not x.mode().empty else None),
    # Diğerleri
    unique_products=('product_id', 'nunique'),
    total_events=('event_time', 'count'),
    session_value=('session_value', 'mean')
)

# One-hot encode 'main_category'
session_df = pd.get_dummies(session_df, columns=['main_category'], prefix='cat')

# --- 3.1 GELİŞMİŞ OTURUM SONRASİ ÖZELLİKLER ---
print("   - Gelişmiş davranış özellikleri hesaplanıyor...")

# Dönüşüm Oranları (Gelişmiş)
session_df['view_to_cart_ratio'] = (session_df['add_to_carts'] / (session_df['views'] + 1)).fillna(0)
session_df['cart_to_buy_ratio'] = (session_df['buys'] / (session_df['add_to_carts'] + 1)).fillna(0) 
session_df['view_to_buy_ratio'] = (session_df['buys'] / (session_df['views'] + 1)).fillna(0)

# Sepet Davranışı
session_df['removal_to_add_ratio'] = (session_df['removals_from_cart'] / (session_df['add_to_carts'] + 1)).fillna(0)
session_df['net_cart_adds'] = session_df['add_to_carts'] - session_df['removals_from_cart']
session_df['is_cart_abandoned'] = ((session_df['add_to_carts'] > 0) & (session_df['buys'] == 0)).astype(int)

# Kullanıcı Segmentasyonu (Gelişmiş)
session_df['is_new_user'] = (session_df['user_session_count'] <= 1).astype(int)
session_df['is_frequent_buyer'] = (session_df['user_buy_count'] >= 3).astype(int)
session_df['user_value_segment'] = pd.cut(session_df['user_avg_session_value'], 
                                         bins=3, labels=['low', 'medium', 'high'])

# Aktivite Yoğunluğu
session_df['events_per_minute'] = session_df['total_events'] / (session_df['session_duration_seconds'] / 60 + 1)
session_df['interaction_diversity'] = (
    (session_df['views'] > 0).astype(int) +
    (session_df['add_to_carts'] > 0).astype(int) +
    (session_df['removals_from_cart'] > 0).astype(int) +
    (session_df['buys'] > 0).astype(int)
)

# Fiyat ve Kategori Özellikleri
session_df['price_sensitivity'] = session_df['avg_inferred_price'] / (session_df['user_avg_session_value'] + 1)
session_df['category_preference_strength'] = session_df['avg_category_popularity'] / global_mean_popularity

# One-hot encode kategorik değişkenler
session_df = pd.get_dummies(session_df, columns=['user_value_segment'], prefix='segment')

session_df.replace([np.inf, -np.inf], 0, inplace=True)

# --- Veriyi Ayırma ve Temizleme ---
train_final = session_df[session_df['session_value'].notna()]
test_final = session_df[session_df['session_value'].isna()]

Q1 = train_final['session_value'].quantile(0.25)
Q3 = train_final['session_value'].quantile(0.75)
IQR = Q3 - Q1
train_final_clean = train_final[(train_final['session_value'] >= Q1 - 1.5 * IQR) & (train_final['session_value'] <= Q3 + 1.5 * IQR)]

X = train_final_clean.drop('session_value', axis=1)
y = train_final_clean['session_value']
X_test_final = test_final.drop('session_value', axis=1)

# Sütunları hizala
X, X_test_final = X.align(X_test_final, join='left', axis=1, fill_value=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# ==============================================================================
# 4. MODELLERİ EĞİTME VE CROSS-VALIDATION İLE KARŞILAŞTIRMA
# ==============================================================================
print("\n--- Adım 4: Modeller Cross-Validation ile Değerlendiriliyor ---")

from sklearn.model_selection import KFold, cross_val_score

def evaluate_model_cv(model, X, y, cv_folds=5):
    """Cross-validation ile model değerlendirme"""
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # RMSE hesaplama
    rmse_scores = -cross_val_score(model, X, y, cv=kfold, 
                                   scoring='neg_root_mean_squared_error', n_jobs=-1)
    
    return rmse_scores.mean(), rmse_scores.std()

# Optimize edilmiş model parametreleri
lgbm_params = {
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000, 
    'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 31, 
    'verbose': -1, 'n_jobs': -1, 'random_state': 42
}

xgb_params = {
    'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05, 
    'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 
    'random_state': 42, 'n_jobs': -1
}

cat_params = {
    'iterations': 1000, 'learning_rate': 0.05, 'depth': 6, 
    'loss_function': 'RMSE', 'verbose': 0, 'random_seed': 42
}

models = {
    "LightGBM": lgb.LGBMRegressor(**lgbm_params),
    "XGBoost": xgb.XGBRegressor(**xgb_params),
    "CatBoost": cb.CatBoostRegressor(**cat_params),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=6, random_state=42)
}

results = []
validation_split = True  # Validation split kullanılsın mı?

if validation_split:
    print("  -> Validation split ile değerlendirme...")
    for name, model in models.items():
        print(f"    - {name} modeli eğitiliyor...")
        
        # Validation split
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        results.append({
            "Model": name, 
            "RMSE": rmse, 
            "R2 Score": r2,
            "Method": "Validation Split"
        })
else:
    print("  -> Cross-validation ile değerlendirme...")
    for name, model in models.items():
        print(f"    - {name} modeli CV ile değerlendiriliyor...")
        
        try:
            rmse_mean, rmse_std = evaluate_model_cv(model, X, y)
            results.append({
                "Model": name,
                "RMSE": rmse_mean,
                "RMSE_Std": rmse_std,
                "Method": "Cross-Validation"
            })
        except Exception as e:
            print(f"      Hata: {e}")

results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
print(f"\n--- Model Karşılaştırma Sonuçları ({'Validation Split' if validation_split else 'Cross-Validation'}) ---")
print(results_df)

# %%
# ==============================================================================
# 5. ENSEMBLE VE FİNAL TAHMİN OLUŞTURMA  
# ==============================================================================
print(f"\n--- Adım 5: Ensemble ve Final Tahminler ---")

# En iyi 3 modeli seç
top_3_models = results_df.head(3)['Model'].tolist()
print(f"Ensemble için seçilen modeller: {top_3_models}")

# Her model için tahmin yap
ensemble_predictions = {}
for model_name in top_3_models:
    model = models[model_name]
    # Tüm temizlenmiş eğitim verisiyle yeniden eğit
    model.fit(X, y)
    predictions = model.predict(X_test_final)
    ensemble_predictions[model_name] = predictions
    print(f"  -> {model_name}: Ortalama tahmin = {predictions.mean():.2f}")

# Basit ortalama ensemble
simple_ensemble = np.mean(list(ensemble_predictions.values()), axis=0)

# Ağırlıklı ensemble (RMSE'ye göre ters ağırlık)
top_rmse_scores = results_df.head(3)['RMSE'].values
weights = 1 / top_rmse_scores
weights = weights / weights.sum()  # Normalize et

weighted_ensemble = np.zeros(len(X_test_final))
for i, model_name in enumerate(top_3_models):
    weighted_ensemble += weights[i] * ensemble_predictions[model_name]
    print(f"  -> {model_name} ağırlığı: {weights[i]:.3f}")

# En iyi tek model tahmini
best_model_name = results_df.iloc[0]["Model"]
best_single_predictions = ensemble_predictions[best_model_name]

# Submission dosyaları oluştur
submissions = {
    'submission_best_single.csv': best_single_predictions,
    'submission_ensemble_simple.csv': simple_ensemble,
    'submission_ensemble_weighted.csv': weighted_ensemble
}

for filename, predictions in submissions.items():
    submission_df = pd.DataFrame({
        'user_session': X_test_final.index, 
        'session_value': np.clip(predictions, 0, None)  # Negatif değerleri sıfıra çek
    })
    submission_df.to_csv(filename, index=False)
    
    print(f"\n'{filename}' oluşturuldu:")
    print(f"  - Ortalama: {predictions.mean():.2f}")
    print(f"  - Min/Max: {predictions.min():.2f} / {predictions.max():.2f}")
    print(f"  - İlk 5 tahmin: {predictions[:5]}")

print(f"\n🎯 ÖNERİ: 'submission_ensemble_weighted.csv' genellikle en iyi sonucu verir!")

# %%
# ==============================================================================
# 6. MODEL ANALİZİ VE ÖZELLİK ÖNEMİ
# ==============================================================================
print("\n--- Adım 6: Model Analizi ve Özellik Önemleri ---")

# En iyi modelin özellik önemini analiz et
best_model_name = results_df.iloc[0]["Model"]

# Modeli tüm veri ile eğit
best_model = models[best_model_name]
best_model.fit(X, y)

try:
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n{best_model_name} - En Önemli 20 Özellik:")
        print(feature_importance.head(20))
        
        # Özellik kategorilerine göre analiz
        feature_categories = {
            'event_counts': ['views', 'add_to_carts', 'removals_from_cart', 'buys'],
            'conversion_ratios': ['view_to_cart_ratio', 'cart_to_buy_ratio', 'view_to_buy_ratio'],
            'user_features': [col for col in X.columns if col.startswith('user_')],
            'time_features': ['session_duration_seconds', 'is_weekend', 'events_per_minute'],
            'product_features': ['unique_products', 'avg_inferred_price'],
            'behavior_features': ['is_cart_abandoned', 'interaction_diversity', 'net_cart_adds']
        }
        
        print(f"\n=== ÖZELLİK KATEGORİSİ ANALİZİ ===")
        for category, features in feature_categories.items():
            category_features = [f for f in features if f in X.columns]
            if category_features:
                category_importance = feature_importance[feature_importance['feature'].isin(category_features)]['importance'].sum()
                print(f"{category}: {category_importance:.4f}")
        
    elif hasattr(best_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(best_model.coef_)
        }).sort_values('importance', ascending=False)
        
        print(f"\n{best_model_name} - En Önemli 20 Özellik (Katsayı Büyüklüğü):")
        print(feature_importance.head(20))
        
except Exception as e:
    print(f"Feature importance hesaplanamadı: {e}")

# Model performans özeti
print(f"\n=== SONUÇ ÖZETİ ===")
print(f"📊 En iyi tek model: {best_model_name}")
print(f"📈 RMSE: {results_df.iloc[0]['RMSE']:.4f}")
if 'R2 Score' in results_df.columns:
    print(f"📈 R² Score: {results_df.iloc[0]['R2 Score']:.4f}")
print(f"🔧 Kullanılan özellik sayısı: {X.shape[1]}")
print(f"📦 Eğitim verisi boyutu: {X.shape[0]} (outlier temizleme sonrası)")
print(f"🎯 Test verisi boyutu: {X_test_final.shape[0]}")

print(f"\n🚀 PERFORMANS İYİLEŞTİRME ÖNERİLERİ:")
print("1. ✅ Ensemble modelleri kullanıldı (weighted ensemble önerilen)")
print("2. ✅ Gelişmiş feature engineering uygulandı")
print("3. ✅ Outlier detection ve temizleme yapıldı")
print("4. 🔄 Hiperparametre optimizasyonu için advanced_ml_utils.py kullanın")
print("5. 🔄 Stacking ensemble için advanced_ml_utils.py kullanın")
print("6. 🔄 Feature selection deneyebilirsiniz")

# Tahmin dağılımları analizi
print(f"\n=== TAHMİN DAĞILIMI ANALİZİ ===")
for filename, predictions in submissions.items():
    clean_name = filename.replace('submission_', '').replace('.csv', '')
    print(f"{clean_name}:")
    print(f"  Ortalama: {predictions.mean():.2f}")
    print(f"  Medyan: {np.median(predictions):.2f}")
    print(f"  Std: {predictions.std():.2f}")
    print(f"  Min/Max: {predictions.min():.2f} / {predictions.max():.2f}")
    print(f"  Negatif tahmin: {(predictions < 0).sum()}")
    print()
