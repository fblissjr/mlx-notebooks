from datasets import load_dataset
import os
import json


def split_dataset(
    dataset_name, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, target_folder=None
):
    """
    Split a HuggingFace dataset into train, validation, and test splits.
    Saves the splits as JSONL files in the ./data folder.

    Args:
        dataset_name (str): Name or path of the HuggingFace dataset.
        train_ratio (float): Ratio of data for the train split.
        val_ratio (float): Ratio of data for the validation split.
        test_ratio (float): Ratio of data for the test split.
        target_folder (str, optional): Folder where the JSONL files should be written. Defaults to None, which uses the default pattern.
    """
    if target_folder is None:
        target_folder = (
            f"./data/{dataset_name.split('/')[0]}_{dataset_name.split('/')[1]}"
        )
    os.makedirs(target_folder, exist_ok=True)

    # debug
    print(target_folder)

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Get the "train" split from the loaded dataset
    full_dataset = dataset["train"]

    # Calculate the number of examples for each split
    num_examples = len(full_dataset)
    train_size = int(num_examples * train_ratio)
    val_size = int(num_examples * val_ratio)
    test_size = num_examples - train_size - val_size

    # Split the dataset
    train_dataset = full_dataset.select(range(train_size))
    val_dataset = full_dataset.select(range(train_size, train_size + val_size))
    test_dataset = full_dataset.select(range(train_size + val_size, num_examples))

    # Save the splits as JSONL files
    train_dataset.to_json(f"{target_folder}/train.jsonl", orient="records", lines=True)
    val_dataset.to_json(f"{target_folder}/valid.jsonl", orient="records", lines=True)
    test_dataset.to_json(f"{target_folder}/test.jsonl", orient="records", lines=True)

    print(f"Dataset '{dataset_name}' split into:")
    print(f"- Train: {len(train_dataset)} examples (data/train.jsonl)")
    print(f"- Validation: {len(val_dataset)} examples (data/valid.jsonl)")
    print(f"- Test: {len(test_dataset)} examples (data/test.jsonl)")


split_dataset("meta-math/MetaMathQA")
