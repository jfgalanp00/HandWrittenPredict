import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import cv2

# Función para cargar los datos de dígitos
def cargar_datos():
    digits = datasets.load_digits()
    return digits

# Función para visualizar muestras de dígitos
def visualizar_muestras(digits):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Entrenamiento: %i" % label)
    return plt.gcf()

# Función para preparar los datos para entrenamiento
def preparar_datos(digits):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

# Función para dividir los datos en entrenamiento, validación y prueba
def dividir_datos(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Función para entrenar el clasificador SVM
def entrenar_clasificador(X_train, y_train):
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    return clf

# Función para realizar predicciones
def predecir(clf, X_test, y_test):
    if st.button('Predecir'):
        indice_aleatorio = np.random.randint(len(X_test))
        imagen_aleatoria = X_test[indice_aleatorio].reshape(8, 8)
        etiqueta_real = y_test[indice_aleatorio]
        st.write(f'Valor a predecir:  {etiqueta_real}')

        # Preparar la imagen para la predicción
        imagen_para_prediccion = X_test[indice_aleatorio]

        # Predecir el número en la imagen
        prediccion = clf.predict([imagen_para_prediccion])

        # Crear una nueva figura
        fig, ax = plt.subplots()
        ax.imshow(imagen_aleatoria, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f'Predicción: {prediccion[0]}')
        ax.axis('off')
        st.pyplot(fig)

# Función para cargar y preprocesar imágenes
def cargar_imagen(uploaded_file):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (8, 8))
    image = cv2.bitwise_not(image)
    image = image.astype(np.float32) / 255.0 * 16
    image = image.reshape((1, -1))
    return image

# Función para predecir con imágenes
def predecir_con_imagen(clf, image):
    prediccion = clf.predict(image)
    return prediccion[0]

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Cargar los datos y mostrar el título de la aplicación
    digits = cargar_datos()
    st.markdown("# Predicción de Números✏️")
    st.sidebar.markdown("✏️  Página Principal ✏️")

    # Visualizar muestras de dígitos
    visualizar_muestras(digits)
    st.pyplot()

    # Preparar datos y dividirlos en conjuntos de entrenamiento, validación y prueba
    data, target = preparar_datos(digits)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(data, target)

    # Entrenar el clasificador SVM
    clf = entrenar_clasificador(X_train, y_train)

    # Predecir con datos aleatorios
    predecir(clf, X_test, y_test)

    # Predecir con imágenes
    st.markdown("## Predecir con imágenes")
    uploaded_file = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cargar_imagen(uploaded_file)
        prediccion = predecir_con_imagen(clf, image)
        st.write(f"Predicción: {prediccion}")

if __name__ == "__main__":
    main()