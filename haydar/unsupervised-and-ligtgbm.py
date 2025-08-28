# %%
# ==============================================================================
# 0. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans # Denetimsiz öğrenme için KMeans
import lightgbm as lgb # Standart model olarak LightGBM'i ekliyoruz
import warnings
import os

# Olası uyarıları bastır
warnings.filterwarnings('ignore')


# %%
# ==============================================================================
# 1. VERİ YÜKLEME VE BİRLEŞTİRME
# ==============================================================================
print("--- Adım 1: Veri Yükleniyor ve Birleştiriliyor ---")

try:
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    print("Veri setleri başarıyla yüklendi.")
except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı. Lütfen 'train.csv' ve 'test.csv' dosyalarının doğru yolda olduğundan emin olun. Detay: {e}")
    exit()

df_full = pd.concat([df_train, df_test], sort=False, ignore_index=True)
print(f"Birleştirilmiş veri seti boyutu: {df_full.shape}")


# %%
# ==============================================================================
# 2. DETAYLI ÖZELLİK MÜHENDİSLİĞİ
# ==============================================================================
print("\n--- Adım 2: Detaylı Özellik Mühendisliği Yapılıyor ---")
df_full['event_time'] = pd.to_datetime(df_full['event_time'])
df_full['day_of_week'] = df_full['event_time'].dt.dayofweek
df_full['hour_of_day'] = df_full['event_time'].dt.hour
df_full['is_weekend'] = df_full['event_time'].dt.dayofweek.isin([5, 6]).astype(int)

user_features = df_train.groupby('user_id').agg(
    user_avg_session_value=('session_value', 'mean'),
    user_session_count=('user_session', 'nunique'),
    user_buy_count=('event_type', lambda x: (x == 'BUY').sum())
).reset_index()
df_full = pd.merge(df_full, user_features, on='user_id', how='left')
df_full['user_avg_session_value'].fillna(df_train['session_value'].mean(), inplace=True)
df_full['user_session_count'].fillna(0, inplace=True)
df_full['user_buy_count'].fillna(0, inplace=True)


# %%
# ==============================================================================
# 3. OTURUM BAZINDA VERİ BİRLEŞTİRME (AGGREGATION)
# ==============================================================================
print("\n--- Adım 3: Veri Oturum Bazında Birleştiriliyor ---")

agg_dict = {
    'views': ('event_type', lambda x: (x == 'VIEW').sum()),
    'add_to_carts': ('event_type', lambda x: (x == 'ADD_CART').sum()),
    'removals_from_cart': ('event_type', lambda x: (x == 'REMOVE_CART').sum()),
    'buys': ('event_type', lambda x: (x == 'BUY').sum()),
    'session_duration_seconds': ('event_time', lambda x: (x.max() - x.min()).total_seconds()),
    'unique_products': ('product_id', 'nunique'),
    'user_avg_session_value': ('user_avg_session_value', 'first'),
    'user_session_count': ('user_session_count', 'first'),
    'user_buy_count': ('user_buy_count', 'first'),
    'session_value': ('session_value', 'mean')
}

session_df = df_full.groupby('user_session').agg(**agg_dict).reset_index()

# Post-Aggregation Features
session_df['view_to_cart_ratio'] = (session_df['add_to_carts'] / session_df['views']).fillna(0)
session_df['cart_to_buy_ratio'] = (session_df['buys'] / session_df['add_to_carts']).fillna(0)
session_df['is_new_user'] = (session_df['user_session_count'] <= 1).astype(int)
session_df.replace([np.inf, -np.inf], 0, inplace=True)
print("Tüm özellikler oturum bazında birleştirildi.")


# %%
# ==============================================================================
# 4. DENETİMSİZ ÖĞRENME İLE OTURUM SEGMENTASYONU
# ==============================================================================
print("\n--- Adım 4: Denetimsiz Öğrenme ile Oturum Segmentasyonu ---")
# Kümeleme için kullanılacak özellikleri seç (hedef değişken hariç)
features_for_clustering = session_df.drop(['user_session', 'session_value'], axis=1, errors='ignore')

# Veriyi ölçeklendir
scaler_cluster = StandardScaler()
features_scaled = scaler_cluster.fit_transform(features_for_clustering)

# K-Means modelini oluştur ve eğit
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
print("K-Means modeli ile oturumlar kümeleniyor...")
session_df['cluster'] = kmeans.fit_predict(features_scaled)

# Küme ID'sini one-hot encoding ile modele uygun hale getir
session_df = pd.get_dummies(session_df, columns=['cluster'], prefix='cluster')
print("Küme bilgisi yeni bir özellik olarak eklendi.")


# %%
# ==============================================================================
# 5. MODELLEME İÇİN VERİ HAZIRLIĞI
# ==============================================================================
print("\n--- Adım 5: Modelleme İçin Veri Hazırlanıyor ---")
train_final = session_df[session_df['session_value'].notna()]
test_final = session_df[session_df['session_value'].isna()]

Q1 = train_final['session_value'].quantile(0.25)
Q3 = train_final['session_value'].quantile(0.75)
IQR = Q3 - Q1
train_final_clean = train_final[(train_final['session_value'] >= Q1 - 1.5 * IQR) & (train_final['session_value'] <= Q3 + 1.5 * IQR)]

X = train_final_clean.drop(['session_value', 'user_session'], axis=1)
y = train_final_clean['session_value']
X_test_final = test_final.drop(['session_value', 'user_session'], axis=1)

# Sütunları hizala
X, X_test_final = X.align(X_test_final, join='left', axis=1, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Ağaç tabanlı modeller için ölçeklendirme zorunlu olmasa da,
# genellikle zararı olmaz ve bazen performansı artırabilir.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_final_scaled = scaler.transform(X_test_final)

print(f"Eğitim seti: {X_train.shape}")


# %%
# ==============================================================================
# 6. LIGHTGBM MODELİNİ OLUŞTURMA VE EĞİTME
# ==============================================================================
print("\n--- Adım 6: LightGBM Modeli Oluşturuluyor ve Eğitiliyor ---")

params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

model = lgb.LGBMRegressor(**params)

print("Model eğitiliyor...")
model.fit(X_train_scaled, y_train,
          eval_set=[(X_val_scaled, y_val)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(100, verbose=True)])


# %%
# ==============================================================================
# 7. FİNAL TAHMİN VE SUBMISSION OLUŞTURMA
# ==============================================================================
print("\n--- Adım 7: Final tahminler yapılıyor ---")
final_predictions = model.predict(X_test_final_scaled)

submission_df = pd.DataFrame({'user_session': test_final.index, 'session_value': final_predictions})
submission_df['session_value'] = submission_df['session_value'].clip(lower=0)
submission_df.to_csv('submission_unsupervised_lgbm_final.csv', index=False)

print("\n'submission_unsupervised_lgbm_final.csv' dosyası başarıyla oluşturuldu.")
print(submission_df.head())
