#%% load dataset preprocessing: normalization,onehotencoding
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D#feature extraction
from tensorflow.keras.layers import Flatten,Dense,Dropout#classification
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import warnings
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

#load cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

class_labels=["Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]
#visualize
fig,axes=plt.subplots(1,5,figsize=(15,10))
for i in range(5):
    axes[i].imshow(x_train[i])
    label=class_labels[int(y_train[i])]
    axes[i].set_title(label)
    axes[i].axis("off")
plt.show()
#veri seti normalizasyonu 
x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255
#one hot encoding 
y_train=to_categorical(y_train,10) #10 class var
y_test=to_categorical(y_test,10)



#%%data augmentation
datagen=ImageDataGenerator(
    rotation_range=20, #20 dereceye kadar dondur
    height_shift_range=0.2, #dikeyde %20 kaydır
    width_shift_range=0.2,#yatayda %20 kaydır
    shear_range=0.2, #goruntu uzerinde kaydır farklı perspektif
    zoom_range=0.2,#zoom yap
    horizontal_flip=True, #simetrigini al 
    fill_mode="nearest" #bos alanları en yakin pixel degeriyle doldur
    
    )
datagen.fit(x_train)


#%%create compile model train model 
# feature ext:Conv->RELU->Conv->Relu->Pooling ->Dropout

model=Sequential()
model.add(Conv2D(32,(3,3),padding="same",activation="relu",input_shape=x_train.shape[1:]))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))#baglantıların yuzde 25 ini rastgele olarak kapat



#feature ext:Conv->Relu->Conv->Relu->Pool->Dropout
model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding="same",activation="relu",input_shape=x_train.shape[1:]))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))#baglantıların yuzde 25 ini rastgele olarak kapat



#classification: Flatten->Dense->Relu->Dropout->(Output Layer)
model.add(Flatten())#vektor olustur
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

#model derleme
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


history=model.fit(datagen.flow(x_train,y_train,batch_size=64),
                       epochs=10,
                       validation_data=(x_test,y_test))

#%%test model and evaluate performance
#modelin test seti uzerinden tahmin yap 
y_pred=model.predict(x_test)
y_pred_class=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_test,axis=1)

#classification report 
report=classification_report(y_true, y_pred_class,target_names=class_labels)
print(report)

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Train loss")
plt.plot(history.history["val_loss"],label="Val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="train accuracy")
plt.plot(history.history["val_accuracy"],label="Val accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("training and validation accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
