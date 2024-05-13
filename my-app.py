import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import pickle
from streamlit_drawable_canvas import st_canvas

# cargar el modelo previamente entrenado y serializado
filename = 'modelo.pickle'
with open(filename, 'rb') as file:
  model = pickle.load(file)

# funcion para preprocesar la imagen
def preprocess_image(image):
    # convertimos la imagen a escala de grises
    image_gray = image.convert('L')
    # reducimos la imgen a 28x28
    image_resized = image_gray.resize((28, 28))
    # conversion a array numpy
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))
    return image_array

# funcion para la prediccion
def predict(image):
    # preprocesamos la imagen
    image_processed = preprocess_image(image)
    # realizamos la prediccion
    prediction = model.predict(image_processed)
    # obtenemos la prediccion
    predicted_class = np.argmax(prediction)
    return predicted_class

# streamlit
st.title('Clasificador de digitos MNIST')
st.write('Seleccione una opcion para realizar la prediccion')

# botones para las posibles opciones de la aplicacion
option = st.radio('', ('Cargar imagen', 'Dibujar'))

# mostramos las dos opciones y cargamos los recursos para cada una
if option == 'Cargar imagen':
    # carga de imagen por el usuario
    uploaded_image = st.file_uploader("Cargar imagen", type=["jpg", "png"])
    if uploaded_image is not None:
      image = Image.open(uploaded_image)
      predicted_digit = predict(image)
      st.write('Predicción:', predicted_digit)

      # mostramos la imagen que se ha cargado
      st.image(uploaded_image, caption='Imagen cargada', width = 200)


elif option == 'Dibujar':
    canvas_size = 200
    canvas = st_canvas(fill_color="black", stroke_width=5, stroke_color="white",
                       background_color="black", width=canvas_size, height=canvas_size)

    # realzamos la prediccion cuando selecciona el boton
    if st.button("Realizar prediccion"):
        digit_image = Image.new("L", (canvas_size, canvas_size), "black")
        draw = ImageDraw.Draw(digit_image)
        digit_image = Image.fromarray((255 * canvas.image_data[:, :, 0]).astype(np.uint8))
        predicted_digit = predict(digit_image)
        st.write('Predicción:', predicted_digit)
