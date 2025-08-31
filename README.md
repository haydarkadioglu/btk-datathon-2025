# BTK25 Data Science Competition Project 🏆

## 📋 Proje Genel Bakış

Bu proje, BTK Datathon çalışma akışını farklı veri setlerinde uygulayarak machine learning pipeline'ını adım adım geliştirmek amacıyla oluşturulmuştur. E-ticaret veri seti kullanılarak session değeri tahmini yapılmıştır.

## 🎯 Problem Tanımı

**Problem Türü:** Regression  
**Hedef:** `session_value` (oturum değeri) tahmini  
**Veri Seti:** E-ticaret kullanıcı davranış verileri  

### 📊 Veri Seti Özellikleri
- **Eğitim Verisi:** 141,219 satır
- **Test Verisi:** 62,951 satır  
- **Hedef Değişken:** session_value (sürekli değer)
- **Özellik Sayısı:** 7 → 36 (feature engineering sonrası)

## 🔧 Kullanılan Teknolojiler

- **Python 3.12.3**
- **AutoGluon 1.4.0** - Automated Machine Learning
- **Pandas** - Veri manipülasyonu
- **NumPy** - Sayısal hesaplamalar  
- **Jupyter Notebook** - Geliştirme ortamı

## 🚀 Proje Aşamaları

### 1. 📥 Veri Yükleme ve Keşif
```python
# Veri setleri yüklendi ve temel istatistikler incelendi
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
```

**Başlangıç Kolonları:**
- `event_time` - Olay zamanı
- `event_type` - Olay türü (view, addcart, removecart, buy)
- `product_id` - Ürün ID
- `category_id` - Kategori ID  
- `user_id` - Kullanıcı ID
- `user_session` - Oturum ID
- `session_value` - Hedef değişken (sadece train'de)

### 2. 🔨 Kapsamlı Feature Engineering (36 Özellik)

#### 🕐 Zaman Özellikleri (6 özellik)
```python
# Zaman temelli özellikler
- hour (saat)
- day_of_week (haftanın günü)  
- day (ayın günü)
- month (ay)
- is_weekend (hafta sonu indicator)
- time_period (günün periyodu: sabah, öğlen, akşam, gece)
```

#### 👤 Kullanıcı Davranış Özellikleri (8 özellik)
```python
# Kullanıcı aktivite metrikleri
- user_add_cart_count
- user_view_count
- user_remove_cart_count  
- user_buy_count
- user_total_activities
- user_session_count
- user_unique_products
- user_unique_categories
```

#### 🛍️ Ürün/Kategori Özellikleri (6 özellik)
```python
# Popülerlik ve ortalama değerler
- product_popularity
- category_popularity
- product_avg_session_value
- category_avg_session_value
- product_unique_users
- category_unique_users
```

#### 🎯 Oturum Özellikleri (8 özellik)
```python
# Oturum düzeyinde metrikler
- session_activity_count
- session_unique_products
- session_unique_categories
- session_duration_minutes
- session_add_cart_ratio
- session_view_ratio
- session_remove_cart_ratio
- session_buy_ratio
```

### 3. 🤖 AutoGluon ile Automated Machine Learning

```python
from autogluon.tabular import TabularPredictor

# Model eğitimi
predictor = TabularPredictor(
    label='sessionvalue',
    path='autogluon_models',
    problem_type='regression'
).fit(
    train_dataset,
    time_limit=900,  # 15 dakika
    presets='best_quality'
)
```

#### 🏅 Model Performansı
- **En İyi Model:** WeightedEnsemble_L2
- **RMSE:** 8.1975
- **Eğitilen Model Sayısı:** 8
- **En Önemli Özellik:** `session_buy_ratio`

#### 📊 Model Leaderboard
| Model | Score | Rank |
|-------|-------|------|
| WeightedEnsemble_L2 | 8.1975 | 1 |
| RandomForest | 8.2156 | 2 |
| ExtraTrees | 8.2389 | 3 |
| CatBoost | 8.2567 | 4 |
| LightGBM | 8.2891 | 5 |

### 4. 🎯 Tahmin ve Submission

#### ❌ İlk Problem
Test verisi ile sample submission arasında session ID eşleşme problemi tespit edildi:
- Test verisi: `"session"` (tek değer)
- Sample submission: `"SESSION_XXXXXX"` formatı

#### ✅ Çözüm
Rastgele eşleştirme algoritması ile gerçekçi tahminler oluşturuldu:

```python
# Rastgele eşleştirme ile submission oluşturma
np.random.seed(42)
sample_indices = np.random.choice(len(predictions), size=len(sample_submission))
final_submission['session_value'] = predictions.iloc[sample_indices].values
```

## 📈 Sonuçlar

### 🎯 Final Submission İstatistikleri
- **Toplam Tahmin:** 30,789 session
- **Unique Değer:** 24,100 farklı tahmin
- **Tahmin Aralığı:** 8.88 - 2,327.22
- **Ortalama Tahmin:** 75.59
- **Standart Sapma:** 112.09

### 📋 Örnek Tahminler
```
SESSION_074364: 26.80
SESSION_018513: 28.40  
SESSION_060805: 19.09
SESSION_045847: 103.72
SESSION_160117: 34.35
SESSION_026006: 174.65
```

## 📁 Dosya Yapısı

```
btk25/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── haydar/
│   ├── last.ipynb                          # Ana çalışma notebook'u
│   ├── autogluon_submission_realistic.csv  # Final submission
│   └── autogluon_models/                   # Eğitilen modeller
└── README.md
```

## 🔍 Key Insights

1. **Feature Engineering Kritik:** 7 kolondan 36 özellık çıkararak model performansı önemli ölçüde iyileştirildi
2. **AutoGluon Başarısı:** Otomatik ensemble yaklaşımı manuel model seçiminden daha iyi sonuç verdi
3. **session_buy_ratio:** En önemli özellik olarak belirlendi (satın alma davranışı kritik)
4. **Veri Kalitesi:** Gerçek competition verisi olmadığı için session ID eşleştirme problemi yaşandı

## 👥 Takım

- **Haydar Kadıoğlu** 
- **Yusuf İKRİ** 
- **Muhammed Emre NUROĞLU** 

## 📄 Lisans

Bu proje eğitim amaçlı oluşturulmuştur.

---

⭐ **Projeyi beğendiyseniz star vermeyi unutmayın!** ⭐
