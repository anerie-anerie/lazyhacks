import streamlit as st
import os
import shutil

# Function to save uploaded files
def save_uploaded_file(uploaded_file, file_name):
    # Define the directory where files will be saved
    savedir = os.path.join(os.getcwd(), 'uploads')  # Use the current working directory and append 'uploads'

    # Create the 'uploads' folder if it doesn't exist
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Save the file with the specified filename
    file_path = os.path.join(savedir, file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

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

# Main function
def main():
    st.title("Upload Pictures of Your Clothes!")

    # Step 1: User specifies the maximum number of images
    max_images = st.number_input(
        "How many images would you like to upload?", min_value=1, value=1)

    # Step 2: Single dropbox for multiple images
    uploaded_files = st.file_uploader(
        "Upload your images here:",
        type=["png", "jpg"],
        accept_multiple_files=True
    )

    # Ensure users don't upload more than the specified number of images
    if uploaded_files:
        # Store image details
        image_details = []

        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"Image {i + 1}")
            st.image(uploaded_file, caption=f"Image {i + 1}", use_column_width=True)

            # Ask for description and category
            description = st.text_input(
                f"Describe the outfit in Image {i + 1}:", 
                key=f"description_{i}"
            )
            category = st.selectbox(
                f"What type of clothing is this for Image {i + 1}?",
                options=["Top", "Bottom", "Accessory"],
                key=f"category_{i}"
            )

            # Save the uploaded file
            file_name = f"image_{i + 1}.png"
            file_path = save_uploaded_file(uploaded_file, file_name)

            # Save details
            image_details.append({
                "path": file_path,
                "description": description,
                "category": category
            })

        # Step 3: Save button functionality
        if len(image_details) == len(uploaded_files):
            if st.button("Save!"):
                # Process the saved images
                st.success("All images and descriptions saved!")

                # Clear folder if needed
                clear_input_folder()

                # Show the Next Page button
                if st.button("Go to Next Page"):
                    st.markdown("[Next Page](#)")  # Replace `#` with the actual link to the next page

if __name__ == "__main__":
    main()