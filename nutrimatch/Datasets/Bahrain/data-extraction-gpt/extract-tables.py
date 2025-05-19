from io import StringIO
from typing import Annotated, Any
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
from typing import Iterable
import pandas as pd
import os
from helpers import encode_image
import instructor
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# Load the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

def md_to_df(data: Any) -> Any:
    """
    Convert markdown table to DataFrame.

    Args:
        data (Any): The markdown table data to be converted.

    Returns:
        Any: The converted DataFrame.

    """
    # Convert markdown to DataFrame
    if isinstance(data, str):
        return (
            pd.read_csv(
            StringIO(data),  # Process data
            sep="|",
            index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .applymap(lambda x: x.strip() if isinstance(x, str) else x)
        )
    return data


# Define the directories
images_directory = "pages"
output_directory = "extracted_data"
output_meta_data_directory = "extracted_meta_data"

# Define the MarkdownDataFrame model
MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    PlainSerializer(lambda df: df.to_markdown()),
    WithJsonSchema(
        {
            "type": "string",
            "description": "The markdown representation of the table, each one should be tidy, do not try to join tables that should be seperate",
        }
    ),
]

# Initialize the instructor client
client = instructor.from_openai(
    OpenAI(api_key=api_key), mode=instructor.function_calls.Mode.MD_JSON
)


class Table(BaseModel):
    """
    Represents a table with a caption and a dataframe.
    
    Attributes:
        caption (str): The caption or title of the table.
        dataframe (MarkdownDataFrame): The dataframe containing the table data.
    """
    
    caption: str
    dataframe: MarkdownDataFrame


def extract_table(url: str) -> Iterable[Table]:
    """
    Extracts a table from an image using the GPT-4 Vision Preview model.

    Args:
        url (str): The URL of the image containing the table.

    Returns:
        Iterable[Table]: An iterable of Table objects representing the extracted table.

    """
    return client.chat.completions.create(
        model="gpt-4-vision-preview",
        response_model=Iterable[Table],
        max_tokens=1800,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract table from image."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )


def process_image(image_file: str, processed_log: list[str] = []):
    """
    Process an image file to extract tables and save them as CSV files.

    Args:
        image_file (str): The name of the image file to process.
        processed_log (list[str], optional): A list of already processed image files. Defaults to [].

    Returns:
        None
    """
    
    # Get the image name
    image_name = image_file.replace(".jpg", "")

    # Get the path to the image file
    image_path = os.path.join(images_directory, image_file)

    # Check if this image has been processed
    if image_file in processed_log:
        print(f"Skipping '{image_file}' as it's already processed.")
        return None

    print(f"Extracting tables from '{image_file}'")

    try:
        # Encode the image
        image = encode_image(image_path)
        url = f"data:image/jpeg;base64,{image}"
        tables = extract_table(url)

        # If tables is empty, print a message
        if not tables:
            print(f"No tables found in '{image_file}'")

        # Save the tables as CSV files
        export = 0
        for table in tables:
            output_table_path = f"{image_name}_table_{export}.csv"
            table.dataframe.to_csv(os.path.join(output_directory, output_table_path))
            print(f"Saved table as '{output_table_path}'")
            # Save the caption to a text file
            caption_path = f"{image_name}_table_{export}_caption.txt"
            with open(os.path.join(output_meta_data_directory, caption_path), "w") as f:
                f.write(table.caption)
            export += 1
            print(f"Saved caption as '{caption_path}'")

        # Log this image as processed
        with open("processed_images.log", "a") as log_file:
            log_file.write(f"{image_file}\n")
        
    except Exception as e:
        # Log the error
        print(f"Error processing '{image_file}': {e}")

def main():
    image_files = os.listdir(images_directory)

    # Read the log file to determine which images have been processed
    if os.path.exists("processed_images.log"):
        with open("processed_images.log", "r") as log_file:
            processed_log = log_file.read().splitlines()
    else:
        processed_log = []

    # Process the images in parallel
    with ThreadPoolExecutor(max_workers=35) as executor:
        futures = {
            executor.submit(process_image, image_file, processed_log): image_file
            for image_file in image_files if image_file.endswith(".jpg")
        }
        # Ensuring all futures are completed
        for future in futures:
            future.result()
        
    print("All images processed.")


if __name__ == "__main__":
    main()
