# Import Libraries 
import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import requests
import os
import google.generativeai as genai

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

#define your api key and gemini model
genai.configure(api_key= st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question,stream=True)
    return response


# Home page
def home():
    local_image_path = "Images/lensify_photo.png"
    st.image(local_image_path, use_column_width=True)
    
    model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
    labels_path = 'landmarks_classifier_asia_V1_label_map.csv'

    # Load labels
    df = pd.read_csv(labels_path)
    labels = dict(zip(df.id, df.name))

    # Image Processing Function
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
        api_key= st.secrets["HERE_API_KEY"]
        base_url = "https://geocode.search.hereapi.com/v1/geocode"
        params = {
            "q": loc,
            "apiKey": api_key
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data and "items" in data and len(data["items"]) > 0:
                first_item = data["items"][0]
                address = first_item.get("address", {}).get("label", "Unknown")
                latitude = first_item.get("position", {}).get("lat", None)
                longitude = first_item.get("position", {}).get("lng", None)
                return address, latitude, longitude
        return "Private Address ⚠️", None, None


    # File Upload
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])

    if img_file is None:
        st.warning("NOTE :- Please Upload Only Landmark Images!")

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

            # Button to get AI response
            input = f"Search for ${prediction}, respond in English with this points\n\n[Popularity out of 10(only value):\nVisitor Count of 2023(only value, you can take approximate value):\nAge(only value, you can take approximate value):\nSignificance:\nHistory:\nArchitecture:\nDeities: (if not available then remove)\nFestivals: (if not available then remove)\nPilgrimage:\nCultural Significance:]"
            submit = st.button("Search For Full Details 🔍")

            if submit and input:
                response = get_gemini_response(input)
                full_response = ""
                for chunk in response:
                    full_response += chunk.text + " "

                st.success(full_response)
                

            try:
                # Get location info
                address, latitude, longitude = get_map(prediction)
                st.success('Address: ' + address)

                #exception handled
                if(prediction == "Shaolin Temple"):
                    address = "GW5P+C4M, Dengfeng Blvd, Deng Feng Shi, Zheng Zhou Shi, He Nan Sheng, China, 471925"
                    latitude = 34.5086
                    longitude = 112.9353
                elif(prediction == "Vivekananda House"):
                    address = "VIVEKANANDA HOUSE, Kamaraj Salai, Marina Beach Road, Triplicane, Chennai, Tamil Nadu 600005"
                    latitude = 13.0495
                    longitude = 80.2803
                elif(prediction == "Tomb of Akbar the Great"):
                    address = "Tomb of Akbar The Great Area, Sikandra, Agra, Uttar Pradesh 282007"
                    latitude = 27.2206
                    longitude = 77.9505

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
                st.warning("No address found!! or Geocode of the location is private")

        except Exception as e:
            st.error(f"Error occurred: {e}")

        # Using the prediction variable in the Google Maps URL
        location = address.replace(" ", "+")  # Replacing spaces with '+' for URL formatting

        # Google Maps Direction
        directions_url = f"https://www.google.com/maps/dir/?api=1&destination={location}"

        # Displaying the hyperlink in Streamlit
        st.subheader(f"[📌Direction to {prediction}]({directions_url})")

# About page
def about():
    st.title("About Us")

    st.subheader(
        "Welcome to Lensify!\n"
        "I am Sourin Mukherjee with my team, dedicated to providing you with accurate name and location of your uploaded image\n"
    )

    # Insert an image from a local file

    sir_image = "Images/sir_image.png"
    st.image(sir_image, use_column_width=True)
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

# Navigation
selected_page = navigation()

if selected_page == "Home":
    home()
elif selected_page == "About":
    about()
elif selected_page == "Contact Us":
    contact_us()
