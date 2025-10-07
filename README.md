CIFAR-10 Image Classification with CNN

Bu proje, **CIFAR-10 veri seti** üzerinde sıfırdan inşa edilmiş bir **Convolutional Neural Network (CNN)** kullanarak görüntü sınıflandırması yapmaktadır.  
Amaç, temel CNN mimarilerini anlamak, manuel hiperparametre ayarlamaları (learning rate, dropout, optimizer vb.) yapmak ve modelin performansını değerlendirmektir.

---

##  Özellikler

- CIFAR-10 veri seti (10 sınıf, 60.000 renkli görüntü)
- Görüntü ön işleme (normalizasyon, one-hot encoding)
- **Data Augmentation** (döndürme, kaydırma, zoom, yatay çevirme)
- CNN mimarisi: Conv2D → MaxPooling → Dropout → Flatten → Dense
- **Manuel hyperparameter tuning** (learning rate, dropout oranı, epoch sayısı)
- Model değerlendirmesi (classification report, accuracy & loss grafikleri)
