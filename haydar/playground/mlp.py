# %%
# ==============================================================================
# 0. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import warnings

# Olası uyarıları bastır
warnings.filterwarnings('ignore')

# %%
# ==============================================================================
# 1. VERİ YÜKLEME VE BİRLEŞTİRME
# ==============================================================================
print("--- Adım 1: Veri Yükleniyor ve Birleştiriliyor ---")

try:
    # Eğitim ve test verilerini yükle
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    print("Veri setleri başarıyla yüklendi.")
except FileNotFoundError as e:
    print(f"Hata: Dosya bulunamadı. Lütfen 'train.csv' ve 'test.csv' dosyalarının doğru yolda olduğundan emin olun. Detay: {e}")
    exit()

# Özellik mühendisliğini hem eğitim hem de test setine tutarlı bir şekilde uygulamak için birleştir
df_full = pd.concat([df_train, df_test], sort=False, ignore_index=True)
print(f"Birleştirilmiş veri seti boyutu: {df_full.shape}")

# %%
# ==============================================================================
# 2. TEMEL ÖZELLİK MÜHENDİSLİĞİ (ZAMAN BAZLI)
# ==============================================================================
print("\n--- Adım 2: Zaman Bazlı Temel Özellikler Oluşturuluyor ---")

# 'event_time' sütununu datetime formatına çevir
df_full['event_time'] = pd.to_datetime(df_full['event_time'])

# Zamana dayalı yeni özellikler oluştur
df_full['day_of_week'] = df_full['event_time'].dt.dayofweek
df_full['hour_of_day'] = df_full['event_time'].dt.hour
df_full['is_weekend'] = df_full['day_of_week'].isin([5, 6]).astype(int)

print("Zaman bazlı özellikler oluşturuldu: 'day_of_week', 'hour_of_day', 'is_weekend'")

# %%
# ==============================================================================
# 3. GELİŞMİŞ ÖZELLİK MÜHENDİSLİĞİ (OTURUM ÖNCESİ)
# ==============================================================================
print("\n--- Adım 3: Gelişmiş Özellik Mühendisliği (Oturum Öncesi Agregasyon) ---")

print("   - Kullanıcı geçmişi özellikleri oluşturuluyor...")
user_features = df_train.groupby('user_id').agg(
    user_avg_session_value=('session_value', 'mean'),
    user_session_count=('user_session', 'nunique'),
    user_buy_count=('event_type', lambda x: (x == 'BUY').sum())
).reset_index()

print("   - Tahmini ürün fiyat seviyeleri hesaplanıyor...")
product_prices = df_train[df_train['event_type'] == 'BUY'].groupby('product_id')['session_value'].mean().rename('inferred_price')
global_mean_price = product_prices.mean()

print("   - Kategori popülerliği hesaplanıyor...")
category_popularity = df_train[df_train['event_type'] == 'BUY']['category_id'].value_counts().rename('category_popularity')
global_mean_popularity = category_popularity.mean()

print("   - Oluşturulan özellikler ana veri setine ekleniyor...")
df_full = pd.merge(df_full, user_features, on='user_id', how='left')
df_full = pd.merge(df_full, product_prices, on='product_id', how='left')
df_full = pd.merge(df_full, category_popularity, on='category_id', how='left')

df_full['user_avg_session_value'].fillna(df_train['session_value'].mean(), inplace=True)
df_full['user_session_count'].fillna(0, inplace=True)
df_full['user_buy_count'].fillna(0, inplace=True)
df_full['inferred_price'].fillna(global_mean_price, inplace=True)
df_full['category_popularity'].fillna(global_mean_popularity, inplace=True)
print("   - Boş değerler dolduruldu.")

# %%
# ==============================================================================
# 4. OTURUM BAZINDA VERİ BİRLEŞTİRME (AGGREGATION)
# ==============================================================================
print("\n--- Adım 4: Veri Oturum Bazında Birleştiriliyor ---")

df_full = pd.get_dummies(df_full, columns=['event_type'], prefix='event')

session_df = df_full.groupby('user_session').agg(
    views=('event_VIEW', 'sum'),
    add_to_carts=('event_ADD_CART', 'sum'),
    removals_from_cart=('event_REMOVE_CART', 'sum'),
    buys=('event_BUY', 'sum'),
    session_duration_seconds=('event_time', lambda x: (x.max() - x.min()).total_seconds()),
    avg_inferred_price=('inferred_price', 'mean'),
    is_weekend=('is_weekend', 'max'),
    user_avg_session_value=('user_avg_session_value', 'first'),
    user_session_count=('user_session_count', 'first'),
    user_buy_count=('user_buy_count', 'first'),
    avg_category_popularity=('category_popularity', 'mean'),
    main_category=('category_id', lambda x: x.mode()[0] if not x.mode().empty else None),
    unique_products=('product_id', 'nunique'),
    total_events=('event_time', 'count'),
    session_value=('session_value', 'mean')
)

session_df = pd.get_dummies(session_df, columns=['main_category'], prefix='cat')
print("Veriler oturum bazında birleştirildi.")

# %%
# ==============================================================================
# 5. OTURUM SONRASI ÖZELLİK MÜHENDİSLİĞİ
# ==============================================================================
print("\n--- Adım 5: Oturum Sonrası Yeni Özellikler Oluşturuluyor ---")

session_df['view_to_cart_ratio'] = (session_df['add_to_carts'] / session_df['views']).fillna(0)
session_df['cart_to_buy_ratio'] = (session_df['buys'] / session_df['add_to_carts']).fillna(0)
session_df['removal_to_add_ratio'] = (session_df['removals_from_cart'] / session_df['add_to_carts']).fillna(0)
session_df['is_cart_abandoned'] = ((session_df['add_to_carts'] > 0) & (session_df['buys'] == 0)).astype(int)
session_df['is_new_user'] = (session_df['user_session_count'] <= 1).astype(int)
session_df.replace([np.inf, -np.inf], 0, inplace=True)
print("Oturum sonrası özellikler oluşturuldu.")

# %%
# ==============================================================================
# 6. MODELLEME İÇİN VERİ HAZIRLIĞI
# ==============================================================================
print("\n--- Adım 6: Modelleme İçin Veri Hazırlanıyor ---")

train_final = session_df[session_df['session_value'].notna()]
test_final = session_df[session_df['session_value'].isna()]

Q1 = train_final['session_value'].quantile(0.25)
Q3 = train_final['session_value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
train_final_clean = train_final[(train_final['session_value'] >= lower_bound) & (train_final['session_value'] <= upper_bound)]
print(f"Aykırı değer temizliği sonrası eğitim verisi boyutu: {train_final_clean.shape}")

X = train_final_clean.drop('session_value', axis=1)
y = train_final_clean['session_value']
X_test_final = test_final.drop('session_value', axis=1)

X, X_test_final = X.align(X_test_final, join='left', axis=1, fill_value=0)
print("Eğitim ve test setlerinin sütunları hizalandı.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Veri Ölçeklendirme (Derin Öğrenme için KRİTİK) ---
print("   - Veri ölçeklendiriliyor...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_final_scaled = scaler.transform(X_test_final)
print(f"Eğitim seti: {X_train_scaled.shape}, Validasyon seti: {X_val_scaled.shape}, Test seti: {X_test_final_scaled.shape}")

# %%
# ==============================================================================
# 7. DERİN ÖĞRENME MODELİNİ OLUŞTURMA VE EĞİTME
# ==============================================================================
print("\n--- Adım 7: Derin Öğrenme Modeli Oluşturuluyor ve Eğitiliyor ---")

# Keras'ın her epoch sonunda RMSE'yi hesaplaması için özel bir metrik fonksiyonu
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# Model mimarisini daha derin olacak şekilde güncelle
model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1) # Regresyon için tek bir çıktı nöronu (aktivasyon fonksiyonu yok)
])

# Modeli derle ve metrikleri ekle
# 'loss' olarak MSE'yi kullanıyoruz ve ek olarak MSE ve RMSE'yi de takip ediyoruz.
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', rmse])

# Erken durdurma callback'i: modelin performansı artmadığında eğitimi durdurur
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğit
# Eğitim sırasında Keras, her epoch için loss, mse, rmse ve bunların validasyon karşılıklarını (val_loss, val_mse, val_rmse) yazdıracaktır.
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100, # Maksimum epoch sayısı
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Model performansını değerlendir
print("\n--- Model Performans Değerlendirmesi ---")
y_pred = model.predict(X_val_scaled).flatten()
final_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"Final Validasyon RMSE: {final_rmse:.4f}")
print(f"Final Validasyon R2 Skoru: {r2:.4f}")

# %%
# ==============================================================================
# 8. FİNAL TAHMİN VE SUBMISSION DOSYASI OLUŞTURMA
# ==============================================================================
print(f"\n--- Adım 8: Final tahminler yapılıyor ---")

# Final tahminler için modeli tüm temizlenmiş veriyle yeniden eğitmek iyi bir pratiktir.
# Ancak, zaman kazanmak için mevcut eğitilmiş en iyi modeli de kullanabiliriz.
# Burada mevcut modeli kullanıyoruz.
print("Test seti için tahminler yapılıyor...")
final_predictions = model.predict(X_test_final_scaled).flatten()

# Submission dosyasını oluştur
submission_df = pd.DataFrame({'user_session': X_test_final.index, 'session_value': final_predictions})
submission_df['session_value'] = submission_df['session_value'].clip(lower=0)
submission_df.to_csv('submission_dl_final.csv', index=False)

print("\n'submission_dl_final.csv' dosyası başarıyla oluşturuldu.")
print("Submission dosyasının ilk 5 satırı:")
print(submission_df.head())
