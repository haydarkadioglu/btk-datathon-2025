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

# --- 3.1 YENİ ÖZELLİKLER (POST-AGGREGATION) ---
print("   - Oturum sonrası yeni özellikler hesaplanıyor...")
# Dönüşüm Oranları
session_df['view_to_cart_ratio'] = (session_df['add_to_carts'] / session_df['views']).fillna(0)
session_df['cart_to_buy_ratio'] = (session_df['buys'] / session_df['add_to_carts']).fillna(0)
# Sepet Davranışı
session_df['removal_to_add_ratio'] = (session_df['removals_from_cart'] / session_df['add_to_carts']).fillna(0)
session_df['is_cart_abandoned'] = ((session_df['add_to_carts'] > 0) & (session_df['buys'] == 0)).astype(int)
# Kullanıcı Segmentasyonu
session_df['is_new_user'] = (session_df['user_session_count'] <= 1).astype(int)

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
# 4. MODELLERİ EĞİTME VE KARŞILAŞTIRMA (PARAMETRE OPTİMİZASYONU İLE)
# ==============================================================================
print("\n--- Adım 4: Modeller Optimize Edilmiş Parametrelerle Eğitiliyor ---")

# Optimize edilmiş başlangıç parametreleri
lgbm_params = {'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42}
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1}
cat_params = {'iterations': 1000, 'learning_rate': 0.05, 'depth': 6, 'loss_function': 'RMSE', 'verbose': 0, 'random_seed': 42}

models = {
    "LightGBM": lgb.LGBMRegressor(**lgbm_params),
    "XGBoost": xgb.XGBRegressor(**xgb_params),
    "CatBoost": cb.CatBoostRegressor(**cat_params),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=6, random_state=42)
}

results = []
for name, model in models.items():
    print(f"  -> {name} modeli eğitiliyor...")
    # DÜZELTME: Her modelin erken durdurma mekanizması farklıdır.
    if name == "LightGBM":
         model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name in ["XGBoost", "CatBoost"]:
         model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    else:
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    results.append({"Model": name, "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)), "R2 Score": r2_score(y_val, y_pred)})

results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
print("\n--- Model Karşılaştırma Sonuçları (Optimize Edilmiş) ---")
print(results_df)

# %%
# ==============================================================================
# 5. FİNAL TAHMİN VE SUBMISSION OLUŞTURMA
# ==============================================================================
best_model_name = results_df.iloc[0]["Model"]
print(f"\n--- Adım 5: En iyi model ({best_model_name}) ile final tahminler yapılıyor ---")

best_model = models[best_model_name]
# DÜZELTME: Tüm temizlenmiş eğitim verisiyle yeniden eğit (erken durdurma olmadan)
best_model.fit(X, y)

final_predictions = best_model.predict(X_test_final)

submission_df = pd.DataFrame({'user_session': X_test_final.index, 'session_value': final_predictions})
submission_df['session_value'] = submission_df['session_value'].clip(lower=0)
submission_df.to_csv('submission_v4.csv', index=False)

print("\n'submission_v4.csv' dosyası başarıyla oluşturuldu.")
print(submission_df.head())
