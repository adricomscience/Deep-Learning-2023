import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Contoh data pelatihan (X) dan label (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Membangun model jaringan saraf
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu')) # Layer input dengan 4 neuron dan fungsi aktivasi ReLU
model.add(Dense(1, activation='sigmoid')) # Layer output dengan 1 neuron dan fungsi aktivasi sigmoid

# Kompilasi model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Melatih model
model.fit(X, y, epochs=1000, verbose=2)

# Menguji model
hasil_prediksi = model.predict(X)
print("Hasil Prediksi:")
print(hasil_prediksi)
