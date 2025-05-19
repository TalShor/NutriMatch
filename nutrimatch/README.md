# Project Title

## Introduction
A brief overview of the project, explaining its purpose and significance.

## Methodology
Explanation of the process:
- Define classes for each food item using Pydantic's BaseModel.
- Map base database items to these classes.
- Generate embeddings using GPT for database items.
- For matching with another database (e.g., HPP):
  - Create corresponding classes for its items.
  - Translate/convert to the base class format.
  - Handle discrepancies by reprocessing items, omitting certain fields if necessary.
  - Create embeddings for this database.
- Utilize cosine similarity for matching items across databases.

## Module Structure
1. database_alignment.py runs the matching process between two chosen databases
2. comparing_database_options.py compares all matches for the HPP registry and chooses the most relevant matches 
3. cross_dataset_mapping.py performs cross-dataset mapping by calculating cosine similarity between embeddings from multiple datasets and identifying top matches for each food item.
4. cross_dataset_mapping_validation.py validates the cross-dataset mapping results by asking ChatGPT to classify each match as correct or incorrect and saves the validated matches for further analysis.

## Adding New Datasets
Steps to integrate new datasets:
- Define a new class for the dataset's items by creating enccessary files in the Datasets folder.
- Detail the conversion or mapping process to the base class.
- Outline the embedding and matching process.

## Usage and Examples
Instructions on how to use the module and examples demonstrating functionality.


## Contributing
Guidelines for contributing to the project.

## License
Information about the project's license.
