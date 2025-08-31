import os
import pandas as pd
import csv
from fastapi import APIRouter, UploadFile, File, HTTPException
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from app.config import settings

router = APIRouter()

def stratified_group_splitting(df):
    """
    Perform stratified group-aware splitting of the dataset into train, validation, and test sets.

    Steps:
    1. Assigns a unique `group_id` for each distinct question, ensuring that all identical questions 
       (and their answers) are treated as a single group when splitting.
    2. Uses StratifiedGroupKFold to split the dataset into 5 folds, maintaining stratification by 
       `question` labels while ensuring that no group (same question) is split across folds.
    3. Selects the first fold as the validation/test pool (20% of data), and the remaining as training set (80%).
    4. Further splits the validation/test pool into equal halves (10% each) using another 
       StratifiedGroupKFold, again ensuring group consistency.
    5. Saves the resulting train, validation, and test splits into separate CSV files under 
       the configured `settings.data_path`.

    Returns:
        tuple: Paths to the saved train, validation, and test CSV files.
    """

    # Assign group_id per unique question
    df["group_id"] = df.groupby("question").ngroup()

    # Prepare StratifiedGroupKFold splits
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Get first fold → train + (val/test pool)
    train_idx, pool_idx = next(
        sgkf.split(df, y=df["question"], groups=df["group_id"])
    )
    train_df = df.iloc[train_idx]
    pool_df = df.iloc[pool_idx]

    # Split pool into val/test (50/50 of 20%)
    sgkf_inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val_idx, test_idx = next(
        sgkf_inner.split(pool_df, y=pool_df["question"], groups=pool_df["group_id"])
    )
    val_df = pool_df.iloc[val_idx]
    test_df = pool_df.iloc[test_idx]

    # Save CSVs
    train_path = os.path.join(settings.data_path, "train.csv")
    val_path = os.path.join(settings.data_path, "val.csv")
    test_path = os.path.join(settings.data_path, "test.csv")

    train_df.to_csv(train_path, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
    val_df.to_csv(val_path, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
    test_df.to_csv(test_path, index=False, quotechar='"', quoting=csv.QUOTE_ALL)

    return train_path, val_path, test_path


def group_common_question_splitting(df):
    """
    Perform grouping and simple splitting of the dataset based on unique questions.

    Steps:
    1. Groups all answers belonging to the same question and merges them into a single record 
       with answers concatenated using ' | ' separator.
    2. Resets the index to create a unique `record_id` for each grouped question and assigns 
       it as the `group_id`.
    3. Randomly splits the dataset into training (80%) and validation (20%) sets, 
       without stratification or test set generation.
    4. Saves the train and validation splits into CSV files under the configured `settings.data_path`.

    Returns:
        tuple: Paths to the saved train and validation CSV files.
    """
    # Assign group_id per unique question
    df = df.groupby("question", as_index=False).agg({"answer": lambda x: " | ".join(map(str, x))})
    df = df.reset_index().rename(columns={'index': 'record_id'})
    df["group_id"] = df["record_id"]

    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)

    # Save CSVs
    train_path = os.path.join(settings.data_path, "train.csv")
    val_path = os.path.join(settings.data_path, "val.csv")

    train_df.to_csv(train_path, index=False, quotechar='"', quoting=csv.QUOTE_ALL)
    val_df.to_csv(val_path, index=False, quotechar='"', quoting=csv.QUOTE_ALL)

    return train_path, val_path



@router.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV with columns: question, answer.
    Adds:
      - record_id: incremental ID
      - group_id: same question → same group
    Splits into train/val/test with nested StratifiedGroupKFold (80/10/10).
    Saves results in /data/ as train.csv, val.csv, test.csv.
    """
    # Ensure data directory exists
    os.makedirs(settings.data_path, exist_ok=True)

    # Check file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        # Read uploaded CSV into DataFrame
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents), quotechar='"')
        df = df.dropna()

        # Validate required columns
        if not {"question", "answer"}.issubset(df.columns):
            raise HTTPException(status_code=400, detail="CSV must contain 'question' and 'answer' columns.")

        # Add record_id
        # df["record_id"] = range(1, len(df) + 1)
        df = df.reset_index().rename(columns={'index': 'record_id'})

        # saved_files = stratified_group_splitting(df)
        saved_files = group_common_question_splitting(df)
        
        return {
            "message": "Upload and split successful.",
            "saved_files": saved_files,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
