import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

# Set page configuration
st.set_page_config(
    page_title="Lensify",
    page_icon="📷",
)

# Navbar
def navigation():
    page_options = ["Home", "About", "Contact Us"]
    selected_page = st.sidebar.radio("Navigation", page_options)
    return selected_page

# Home page
def home():
    st.title("Landmark Recognition")

    model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
    labels_path = 'landmarks_classifier_asia_V1_label_map.csv'

    # Load labels
    df = pd.read_csv(labels_path)
    labels = dict(zip(df.id, df.name))

    def image_processing(image):
        img_shape = (321, 321)
        classifier = hub.load(model_url)

        # Extract the specific signature we need
        signature = classifier.signatures['default']

        # Resize and normalize image
        img = tf.image.resize(image, img_shape) / 255.0

        # Predict
        result = signature(tf.expand_dims(img, axis=0))
        prediction = labels[np.argmax(result['predictions:logits'][0])]

        # Convert the image to PIL format for display
        img_pil = PIL.Image.fromarray((image * 255).astype(np.uint8))

        return prediction, img_pil

    def get_map(loc):
        geolocator = Nominatim(user_agent="Your_Name")
        location = geolocator.geocode(loc)
        return location.address, location.latitude, location.longitude

    # File uploader for image selection
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])

    if img_file is not None:
        try:
            # Open and preprocess the uploaded image
            img = PIL.Image.open(img_file)
            img_array = np.array(img)

            # Process image
            prediction, img_copy = image_processing(img_array)

            # Display the uploaded image and the processed image side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_file, caption='Uploaded Image', use_column_width=True)
            with col2:
                st.image(img_copy, caption='Processed Image', use_column_width=True)

            st.header("Predicted Landmark is - ")
            st.success(prediction)

            try:
                # Get location info
                address, latitude, longitude = get_map(prediction)
                st.success('Address: ' + address)

                # Display latitude and longitude
                loc_dict = {'Latitude': latitude, 'Longitude': longitude}
                st.subheader('✅ **Latitude & Longitude of ' + prediction + '**')
                st.json(loc_dict)

                # Display the location on the map
                data = [[latitude, longitude]]
                df = pd.DataFrame(data, columns=['lat', 'lon'])
                st.subheader('✅ **' + prediction + ' on the Map**' + '🗺️')
                st.map(df)

            except Exception as e:

                # If location not found, try again with the last element of the prediction
                prediction_elements = prediction.split(", ")
                new_prediction = prediction_elements[-1] if len(prediction_elements) > 1 else prediction
                try:
                    address, latitude, longitude = get_map(new_prediction)
                    st.success('Address: ' + address)
                    loc_dict = {'Latitude': latitude, 'Longitude': longitude}
                    st.subheader('✅ **Latitude & Longitude of ' + new_prediction + '**')
                    st.json(loc_dict)

                    # Display the location on the map
                    data = [[latitude, longitude]]
                    df = pd.DataFrame(data, columns=['lat', 'lon'])
                    st.subheader('✅ **' + new_prediction + ' on the Map**' + '🗺️')
                    st.map(df)

                except Exception as e:
                    st.warning("No address found for the alternative prediction!")

        except Exception as e:
            st.error(f"Error occurred: {e}")

# About page
def about():
    st.title("About Us")

    st.write(
        "1. Welcome to Lensify!\n"
        "2. I am Sourin Mukherjee with my team, dedicated to providing you with accurate name and location of your uploaded image\n"
    )

    # Insert an image from a local file
    team_image = "Images/team.png"
    st.image(team_image, use_column_width=True)

    st.success("Thank you for choosing our Weather App!")

# Contact Us page
def contact_us():
    st.title(":mailbox: Get In Touch With Us")
    st.write("Feel free to reach out to us with any questions, feedback, or inquiries.")

    contact_form = """
    <form action="https://formsubmit.co/sourin.mukherjee2105833@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="text" name="name" placeholder="Your name" required>
    <input type="email" name="email" placeholder="Your email" required>
    <textarea name="message" placeholder="Your message here"></textarea>
    <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)


    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f" <style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style.css")

# Main application logic
selected_page = navigation()

if selected_page == "Home":
    home()
elif selected_page == "About":
    about()
elif selected_page == "Contact Us":
    contact_us()
