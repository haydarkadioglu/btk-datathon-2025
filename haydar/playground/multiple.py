
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
# 1. VERİ YÜKLEME VE TEMEL İŞLEME (AYRI TRAIN VE TEST)
# ==============================================================================
print("--- Adım 1: Veri Yükleniyor ve Temel Özellikler Oluşturuluyor ---")

try:
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı. Detay: {e}")
    exit()

for df in [df_train, df_test]:
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# %%
# ============================================================================== 
# 2. ÖZELLİK MÜHENDİSLİĞİ (TRAIN ÜZERİNDE)
# ==============================================================================
print("--- Adım 2: Train üzerinde özellik mühendisliği ---")

# Kullanıcı geçmişi özellikleri
user_features = df_train.groupby('user_id').agg(
    user_avg_session_value=('session_value', 'mean'),
    user_session_count=('user_session', 'nunique'),
    user_buy_count=('event_type', lambda x: (x == 'BUY').sum())
).reset_index()

# Tahmini ürün fiyatı
product_prices = df_train[df_train['event_type'] == 'BUY'].groupby('product_id')['session_value'].mean().rename('inferred_price')
global_mean_price = product_prices.mean()

# Kategori popülerliği
category_popularity = df_train[df_train['event_type'] == 'BUY']['category_id'].value_counts().rename('category_popularity')
global_mean_popularity = category_popularity.mean()

# Train verisine ekle
df_train = pd.merge(df_train, user_features, on='user_id', how='left')
df_train = pd.merge(df_train, product_prices, on='product_id', how='left')
df_train = pd.merge(df_train, category_popularity, on='category_id', how='left')

df_train['user_avg_session_value'].fillna(df_train['session_value'].mean(), inplace=True)
df_train['user_session_count'].fillna(0, inplace=True)
df_train['user_buy_count'].fillna(0, inplace=True)
df_train['inferred_price'].fillna(global_mean_price, inplace=True)
df_train['category_popularity'].fillna(global_mean_popularity, inplace=True)

# %%
# ============================================================================== 
# 3. TEST VERİSİNE ÖZELLİKLERİN UYGULANMASI
# ==============================================================================
print("--- Adım 3: Test verisine train’den öğrenilen özellikler uygulanıyor ---")

df_test = pd.merge(df_test, user_features, on='user_id', how='left')
df_test = pd.merge(df_test, product_prices, on='product_id', how='left')
df_test = pd.merge(df_test, category_popularity, on='category_id', how='left')

df_test['user_avg_session_value'].fillna(df_train['session_value'].mean(), inplace=True)
df_test['user_session_count'].fillna(0, inplace=True)
df_test['user_buy_count'].fillna(0, inplace=True)
df_test['inferred_price'].fillna(global_mean_price, inplace=True)
df_test['category_popularity'].fillna(global_mean_popularity, inplace=True)

# %%
# ============================================================================== 
# 4. OTURUM BAZINDA AGGREGATION (TRAIN & TEST AYRI)
# ==============================================================================
print("--- Adım 4: Oturum bazında veri birleştirme ---")

def aggregate_sessions(df, is_train=True):
    df = pd.get_dummies(df, columns=['event_type'], prefix='event')
    
    agg_dict = {
        'views': ('event_VIEW', 'sum'),
        'add_to_carts': ('event_ADD_CART', 'sum'),
        'removals_from_cart': ('event_REMOVE_CART', 'sum'),
        'buys': ('event_BUY', 'sum'),
        'session_duration_seconds': ('event_time', lambda x: (x.max() - x.min()).total_seconds()),
        'avg_inferred_price': ('inferred_price', 'mean'),
        'is_weekend': ('is_weekend', 'max'),
        'user_avg_session_value': ('user_avg_session_value', 'first'),
        'user_session_count': ('user_session_count', 'first'),
        'user_buy_count': ('user_buy_count', 'first'),
        'avg_category_popularity': ('category_popularity', 'mean'),
        'main_category': ('category_id', lambda x: x.mode()[0] if not x.mode().empty else None),
        'unique_products': ('product_id', 'nunique'),
        'total_events': ('event_time', 'count')
    }

    if is_train:
        agg_dict['session_value'] = ('session_value', 'mean')

    session_df = df.groupby('user_session').agg(**agg_dict)
    session_df = pd.get_dummies(session_df, columns=['main_category'], prefix='cat')

    session_df['view_to_cart_ratio'] = (session_df['add_to_carts'] / session_df['views']).fillna(0)
    session_df['cart_to_buy_ratio'] = (session_df['buys'] / session_df['add_to_carts']).fillna(0)
    session_df['removal_to_add_ratio'] = (session_df['removals_from_cart'] / session_df['add_to_carts']).fillna(0)
    session_df['is_cart_abandoned'] = ((session_df['add_to_carts'] > 0) & (session_df['buys'] == 0)).astype(int)
    session_df['is_new_user'] = (session_df['user_session_count'] <= 1).astype(int)
    
    session_df.replace([np.inf, -np.inf], 0, inplace=True)
    return session_df

# Train ve test ayrı çağır
train_sessions = aggregate_sessions(df_train, is_train=True)
test_sessions = aggregate_sessions(df_test, is_train=False)


# Temizlenmiş train
Q1 = train_sessions['session_value'].quantile(0.25)
Q3 = train_sessions['session_value'].quantile(0.75)
IQR = Q3 - Q1
train_clean = train_sessions[(train_sessions['session_value'] >= Q1 - 1.5 * IQR) & (train_sessions['session_value'] <= Q3 + 1.5 * IQR)]
# Temizlenmiş train verisi
X = train_clean.drop('session_value', axis=1)
y = train_clean['session_value']

# Test verisi
X_test_final = test_sessions.copy()  # session_value yok, direkt kopya alıyoruz

# Train ve test sütunlarını hizala, eksik olanları 0 ile doldur
X, X_test_final = X.align(X_test_final, join='left', axis=1, fill_value=0)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# ============================================================================== 
# 5. MODELLERİ EĞİTME VE KARŞILAŞTIRMA (PARAMETRE OPTİMİZASYONU İLE)
# ==============================================================================
print("\n--- Adım 5: Modeller Optimize Edilmiş Parametrelerle Eğitiliyor ---")

# Optimize edilmiş parametreler
lgbm_params = {'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 1000, 'learning_rate': 0.05,
               'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.1,
               'lambda_l2': 0.1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42}
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1}
cat_params = {'iterations': 1000, 'learning_rate': 0.05, 'depth': 6, 'loss_function': 'RMSE', 'verbose': 0,
              'random_seed': 42}

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
    if name == "LightGBM":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name in ["XGBoost", "CatBoost"]:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    results.append({"Model": name, "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)), "R2 Score": r2_score(y_val, y_pred)})

results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
print("\n--- Model Karşılaştırma Sonuçları ---")
print(results_df)

# %%
# ============================================================================== 
# 6. FİNAL TAHMİN VE SUBMISSION (%50 CatBoost + %50 LightGBM)
# ==============================================================================
print("\n--- Adım 6: CatBoost ve LightGBM ortalaması ile final tahminler ---")

# Model nesneleri
cat_model = cb.CatBoostRegressor(**cat_params)
lgb_model = lgb.LGBMRegressor(**lgbm_params)

# Tüm temizlenmiş eğitim verisiyle yeniden eğit
cat_model.fit(X, y, verbose=False)
lgb_model.fit(X, y)

# Tahminler
cat_preds = cat_model.predict(X_test_final)
lgb_preds = lgb_model.predict(X_test_final)

# Ortalama tahmin
final_predictions = 0.5 * cat_preds + 0.5 * lgb_preds

# Submission
submission_df = pd.DataFrame({'user_session': X_test_final.index, 'session_value': final_predictions})
submission_df['session_value'] = submission_df['session_value'].clip(lower=0)
submission_df.to_csv('submission_v_final.csv', index=False)

print("\n'submission_v_final.csv' dosyası başarıyla oluşturuldu.")
print(submission_df.head())




# %%
