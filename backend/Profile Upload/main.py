import streamlit as st
import os
import shutil
from PIL import Image

# Load the image
img = Image.open('background.png')

#import functions

# Function to save uploaded files
def save_uploaded_file(uploaded_file, label):
    # Define the directory where files will be saved, relative to the project folder
    savedir = os.path.join(os.getcwd(), 'uploads')  # Use the current working directory and append 'uploads'

    # Create the 'uploads' folder if it doesn't exist
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Save the file with the label as the filename
    file_path = os.path.join(savedir, f"{label}.png")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


# Function to create the upload interface for images
def image_dropboxes():
    labels = ["personal picture"]
    image_files = {}

    for label in labels:
        file = st.file_uploader(label=f"Upload your {label} here:", type=["png", "jpg"])

        if file:
            # Display the uploaded image
            st.image(file, caption=label)

            # Save the file immediately with the appropriate label
            file_path = save_uploaded_file(file, label.split()[0])
            image_files[label] = file_path

    return image_files

# Function to clear all images in uploads folder
def clear_input_folder():
    # Get the absolute path to the folder within the current workspace
    folder_path = os.path.join(os.getcwd(), "uploads")
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
    
    # Loop through all files in the folder and remove them
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
        else:
            print(f"Skipped: {file_path} (not a file)")

    print("Clearing completed.")

# Inject custom CSS to style the page
st.markdown("""
    <style>

        /* Custom title styling */
        h1 {
            color: #000000;
            font-size: 50px;
            font-family: 'Verdana', sans-serif;
        }

        /* Custom file uploader box styling */
        .css-1wa3eu0 {
            background-color: #eff6f9;
            border: 2px solid #016d77;
            border-radius: 10px;
            padding: 15px;
            font-size: 18px;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #eff6f9;
            color: black;
            border-radius: 12px;
            font-size: 18px;
            padding: 10px 24px;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #016d77;
            color: white;
        }

        .stButton>button:active {
            background-color: rgba(0, 0, 0, 0);
            color: rgba(0, 0, 0, 0)
        }

        .stButton>button:focus {
            background-color: transparent;
            color: transparent;
            border: transparent;
        }

        /* Image caption styling */
        .stImage caption {
            font-size: 16px;
            color: #000000;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# Main function
def main():
    # Collect uploaded images
    st.title("Create Your Avatar!")
    image_files = image_dropboxes()

    # Only show the button if the image is uploaded
    if len(image_files) == 1:
        if 'button_clicked' not in st.session_state:  # Check if the button was clicked
            st.session_state.button_clicked = False
        
        if not st.session_state.button_clicked:
            if st.button("Save!"):
                # Process the uploaded image if needed
                profile_img = image_files["personal picture"]

                # Call the anemicScore function with these paths if needed
                # hair_color = predict_haircolor(profile_img)
                # hair_type = predict_hairtype(profile_img)
                # skin_color = predict_skincolor(profile_img)


                # Clear the InputImg folder after the calculation
                clear_input_folder()
                st.info("Saved!")

                if st.button("Go to Next Page"):
                    st.markdown("[Next Page](#)")  # You can replace `#` with the actual link

if __name__ == "__main__":
    main()