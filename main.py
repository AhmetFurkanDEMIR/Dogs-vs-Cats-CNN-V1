from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
from time import sleep as sl
from keras import layers
from keras import models
import os, shutil


# ----------- veri ön işleme (veri klasör yollarını kodda tanımlama ve seti parçalama) -----------

original_dataset_dir = os.path.join("traina")

train_dir = os.path.join("train") # eğitim seti

validation_dir = os.path.join("validation") # doğrulama seti

test_dir = os.path.join("test") # test seti

train_cats_dir = os.path.join(train_dir,"cats") # kedi eğitim setinin dizini.

train_dogs_dir = os.path.join(train_dir,"dogs") # köpek eğitim seti

validation_cats_dir = os.path.join(validation_dir,"cats") # kedi doğrulama setinin dizini.

validation_dogs_dir = os.path.join(validation_dir,"dogs") # köpek doğrulama seti

test_cats_dir = os.path.join(test_dir,"cats") # kedi test setinin dizini.

test_dogs_dir = os.path.join(test_dir,"dogs") # köpek test seti


# kedi ler için dataseti üç parçaya ayırdık. (eğitim, doğrulama, test)

fnames = ["cat.{}.jpg".format(i) for i in range(1000)]

for fname in fnames:

	src = os.path.join(original_dataset_dir, fname) 

	dst = os.path.join(train_cats_dir, fname) # train_cats_dir

	shutil.copyfile(src,dst) # dizinine kopyalar.



fnames = ["cat.{}.jpg".format(i) for i in range(1000,1500)]

for fname in fnames:

	src = os.path.join(original_dataset_dir, fname) 

	dst = os.path.join(validation_cats_dir, fname) # validation_cats_dir

	shutil.copyfile(src,dst) # dizinine kopyalar.


fnames = ["cat.{}.jpg".format(i) for i in range(1500,2000)]

for fname in fnames:

	src = os.path.join(original_dataset_dir, fname) 

	dst = os.path.join(test_cats_dir, fname) # validation_cats_dir

	shutil.copyfile(src,dst) # dizinine kopyalar.


# kedi ler için dataseti üç parçaya ayırdık. (eğitim, doğrulama, test)

fnames = ["dog.{}.jpg".format(i) for i in range(1000)]

for fname in fnames:

	src = os.path.join(original_dataset_dir, fname) 

	dst = os.path.join(train_dogs_dir, fname) # train_dogs_dir

	shutil.copyfile(src,dst) # dizinine kopyalar.



fnames = ["cat.{}.jpg".format(i) for i in range(1000,1500)]

for fname in fnames:

	src = os.path.join(original_dataset_dir, fname) 

	dst = os.path.join(validation_dogs_dir, fname) # validation_dogs_dir

	shutil.copyfile(src,dst) # dizinine kopyalar.


fnames = ["cat.{}.jpg".format(i) for i in range(1500,2000)]

for fname in fnames:

	src = os.path.join(original_dataset_dir, fname) 

	dst = os.path.join(test_dogs_dir, fname) 

	shutil.copyfile(src,dst) # dizinine kopyalar.


os.system("clear")

print("\n  toplam eğitim - kedi sayısı = ",len(os.listdir(train_cats_dir)))

print("  toplam doğrulama - kedi sayısı = ",len(os.listdir(validation_cats_dir)))

print("  toplam eğitim - köpek sayısı = ",len(os.listdir(train_dogs_dir)))

print("  toplam doğrulama - köpek sayısı = ",len(os.listdir(validation_dogs_dir)))
print()

sl(7)

# ----------- CNN ağı oluşturma -----------

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3))) # 32 birimlik 3,3 filitreli katman

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation="relu"))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation="relu"))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation="relu"))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation="relu"))

model.add(layers.Dense(1, activation="sigmoid"))

# ----------- modeli derleme -----------

model.compile(loss="binary_crossentropy",
			  optimizer = optimizers.RMSprop(lr=1e-4),
			  metrics=["acc"])

# ----------- veri ön işleme (verileri vektör haline getirme) -----------

train_datagen = ImageDataGenerator(rescale=1./255) # eğitim tüm resimleri 1/255 e ölçekler

test_datagen = ImageDataGenerator(rescale=1./255)  # test tüm resimleri 1/255 e ölçekler

train_generator = train_datagen.flow_from_directory(train_dir, # hedef dizin
													target_size=(150,150), # tüm resimler 150x150 boyutuna getirilir.
													batch_size=20,
													class_mode="binary") # binary_crossentropy kullandığımız için ikili etiketler gerekiyor.


validation_generator = test_datagen.flow_from_directory(validation_dir,
													target_size=(150,150), 
													batch_size=20,
													class_mode="binary")


os.system("clear")
print()

for data_batch, labels_batch in train_generator:
	
	print("  veri toplu şekli = ",data_batch.shape)

	print("  etiketler toplu şekli = ",labels_batch.shape)

	break

sl(5)

os.system("clear")
print()

# ----------- eğitim -----------

history = model.fit_generator(train_generator,
							  steps_per_epoch=100,
							  epochs=30,
							  validation_data=validation_generator,
							  validation_steps=50)


model.save("asd.h5")


# ----------- sonucları görselleştirme (modelin verimi) -----------

acc = history.history["acc"]

val_acc = history.history["val_acc"]

loss = history.history["loss"]

val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Eğitim başarımı")

plt.plot(epochs, val_acc, "b", label="Doğrulama başarımı")

plt.title("Eğitim ve doğrulama başarımı")

plt.legend()

plt.figure()

plt.plot(epochs, loss, "bo", label="Eğitim kaybı")

plt.plot(epochs, val_loss, "b", label="Doğrulama kaybı")

plt.title("Eğitim ve doğrulama kaybı")

plt.legend()

plt.show()