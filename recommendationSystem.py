import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o conjunto de dados CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar os valores dos pixels para a faixa de 0 a 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Verificar os dados
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Construir e treinar o modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', name='dense'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Extrair características das imagens usando o modelo treinado
feature_extractor = models.Model(inputs=model.input, outputs=model.get_layer("dense").output)

train_features = feature_extractor.predict(train_images)
test_features = feature_extractor.predict(test_images)

# Calcular similaridade entre imagens
similarity_matrix = cosine_similarity(test_features)

def plot_similarity_matrix(matrix, labels, num_images=10):
    plt.figure(figsize=(10, 10))
    sns.heatmap(matrix[:num_images, :num_images], annot=True, fmt=".2f", cmap='coolwarm', xticklabels=labels[:num_images], yticklabels=labels[:num_images])
    plt.show()

plot_similarity_matrix(similarity_matrix, test_labels.flatten(), 10)

# Criar uma função de recomendação
def recommend_similar_images(image_index, features, num_recommendations=5):
    similarities = cosine_similarity(features[image_index].reshape(1, -1), features).flatten()
    similar_indices = similarities.argsort()[-num_recommendations-1:-1][::-1]
    return similar_indices

def display_similar_images(image_index, similar_indices):
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(similar_indices):
        plt.subplot(1, len(similar_indices), i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[idx], cmap=plt.cm.binary)
        plt.xlabel(class_names[test_labels[idx][0]])
    plt.show()

image_index = 0
similar_indices = recommend_similar_images(image_index, test_features)
display_similar_images(image_index, similar_indices)
