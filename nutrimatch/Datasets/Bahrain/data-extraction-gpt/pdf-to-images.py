from pdf2image import convert_from_path

# brew install poppler

def pdf_to_images(pdf_path, output_folder):
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path, dpi=100)  # dpi can be adjusted based on the desired quality

    # Save each page as a JPG image
    for i, image in enumerate(images):
        image_path = f"{output_folder}/page_{i + 1}.jpg"  # Change the file extension to .jpg
        image.save(image_path, 'JPEG')  # Save the image as JPEG format
        print(f"Saved page {i + 1} as '{image_path}'")

# Specify your PDF path and output folder
pdf_path = "foods.pdf"
output_folder = "pages"

pdf_to_images(pdf_path, output_folder)
