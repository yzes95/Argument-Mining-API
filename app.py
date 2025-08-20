# --- Core Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
import joblib
import spacy
import numpy as np
from scipy.sparse import hstack, csr_matrix
from typing import Literal
from transformers import (
    BertTokenizer,
    BertModel,
    Trainer,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
)
import pandas as pd
import torch
from xaif import AIF as aif
from datasets import Dataset
import re

# =============================================================================
# SECTION 1: TF_IDF_FE HELPER FUNCTIONS
# =============================================================================

# --- Configuration ---
# TODO: Update these file paths to match saved files.
CLASSIC_SVM_MODEL_PATH = "models/tf_idf_svm_argument_classifier_cv_8_gamma_combined.joblib"
TFIDF_VECTORIZER_PATH = "models/tf_idf_vectorizer_combined.joblib"

# --- Initialization ---
# Load spaCy for sentence segmentation
nlp = spacy.load("en_core_web_sm")

# ---Phase 2a: Classic SVM ---
try:
    classic_svm_model = joblib.load(CLASSIC_SVM_MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    print("Classic SVM model and TF-IDF vectorizer loaded.")
except FileNotFoundError:
    classic_svm_model = None
    tfidf_vectorizer = None
    print(
        "WARNING: Classic SVM or vectorizer not found. The 'classic_svm' model will not be available."
    )


# --- Feature Extraction Functions ---
def extract_ngram_features(processed_data):
    ngram_features = joblib.load(TFIDF_VECTORIZER_PATH).transform(processed_data)
    return ngram_features


def extract_lexicon_features(processed_data: pd.DataFrame):
    ARGUMENTATIVE_INDICATORS = {
        "because",
        "since",
        "for",
        "as",
        "therefore",
        "thus",
        "consequently",
        "so",
        "should",
        "must",
        "ought",
        "clearly",
        "evidence",
        "proves",
        "demonstrates",
        "reason",
    }

    Argumentative_statments_count = list()
    count = 0
    processed_data = processed_data.to_dict("records")
    for segment in processed_data:
        words = segment["Text"].lower().split()
        for word in words:
            if word in ARGUMENTATIVE_INDICATORS:
                count += 1
        Argumentative_statments_count.append(count)
        count = 0
    return np.array(Argumentative_statments_count).reshape(-1, 1)


def extract_structural_features(processed_data: pd.DataFrame):
    Features_List = list()
    processed_data = processed_data.to_dict("records")
    for segement in processed_data:
        features = {
            "seg_char_len": len(segement["Text"]),
            "seg_words_count": len(segement["Text"].split()),
        }
        Features_List.append(features)
    return pd.DataFrame(Features_List).to_numpy()


def extract_syntactic_features(processed_data: pd.DataFrame):
    processed_data = processed_data.to_dict("records")
    Features_List = list()
    count_of_nouns = 0
    count_of_verbs = 0
    count_of_adjec = 0
    for segement in processed_data:
        doc = nlp(segement["Text"])
        for token in doc:
            if token.pos_ == "NOUN":
                count_of_nouns += 1
            elif token.pos_ == "VERB":
                count_of_verbs += 1
            elif token.pos_ == "ADJ":
                count_of_adjec += 1
        features = {
            "count_of_nouns": count_of_nouns,
            "count_of_verbs": count_of_verbs,
            "count_of_adjec": count_of_adjec,
        }
        Features_List.append(features)
        count_of_nouns = 0
        count_of_verbs = 0
        count_of_adjec = 0
    return pd.DataFrame(Features_List).to_numpy()


# =============================================================================
# =============================================================================

# =============================================================================
# SECTION 2: BERT HELPER FUNCTIONS
# =============================================================================
# The name of the pre-trained BERT model we will use from Hugging Face.
BERT_MODEL_NAME = "bert-base-uncased"
BERT_SVM_MODEL_PATH = "models/bert_svm_argument_classifier_cv_8_gamma_combined.joblib"


def model_initializer(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return (tokenizer, model, device)


def generate_bert_embeddings(corpus, tokenizer, model, device):
    embeddings_list = []
    print("Generating embeddings... (This may take a while)")
    for text in corpus:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # 11. Append the resulting numpy array to   `embeddings_list`.
        embeddings_list.append(cls_embedding)
    return np.vstack(embeddings_list)


# --- Load components for Phase 2b: Hybrid BERT+SVM ---
try:
    hybrid_svm_model = joblib.load(BERT_SVM_MODEL_PATH)
    tokenizer, model, device = model_initializer(BERT_MODEL_NAME)
    print("Hybrid SVM and BERT components loaded.")
except FileNotFoundError:
    hybrid_svm_model = None
    print(
        "WARNING: Hybrid SVM model not found. The 'hybrid_svm' model will not be available."
    )

# =============================================================================
# =============================================================================

# =============================================================================
# SECTION 3: RoBERTa HELPER FUNCTIONS
# =============================================================================
ROBERTA_MODEL_PATH = "models/roberta_argument_classifier_combined_cv_8"  # Use the final output directory from   training script
try:
    roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
    roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH)
    roberta_model.to(device)
    roberta_model.eval()
    print("Fine-tuned RoBERTa model loaded successfully.")
except OSError:
    roberta_model = None
    roberta_tokenizer = None
    print(
        "WARNING: Fine-tuned RoBERTa model not found. The 'roberta' model will not be available."
    )
# =============================================================================
# =============================================================================


# =============================================================================
# SECTION 4: Pydantic Models (Defining the API's "Data Contracts") and FastAPI app Loading
# =============================================================================
# EXPLANATION: These classes define the exact structure of the data   API
# expects to receive (AnalysisRequest) and the data it will send back (AnalysisResponse).
# FastAPI uses these for automatic data validation and generating interactive documentation.
class TextInput(BaseModel):
    txt: str


class OVAInput(BaseModel):
    # Define fields for OVA if needed, can be kept simple if not used
    firstname: str = ""
    surname: str = ""
    url: str = ""
    nodes: List[Any]
    edges: List[Any]


class AIFInput(BaseModel):
    # Defines the expected structure of the AIF object in the request
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    schemefulfillments: List[Any]
    descriptorfulfillments: List[Any]
    participants: List[Any]
    locutions: List[Any]


class AIFRequest(BaseModel):
    """Defines the structure of the incoming request JSON body."""

    AIF: AIFInput
    text: TextInput
    dialog: bool
    OVA: OVAInput
    model_name: Literal["classic_svm", "bert_svm", "roberta"] = Field(
        default="classic_svm",
        description="Choose 'classic_svm' for manual features or 'bert_svm' for BERT features. or 'roberta' for whole roberta all in all",
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "AIF": {
                    "nodes": [],
                    "edges": [],
                    "schemefulfillments": [],
                    "descriptorfulfillments": [],
                    "participants": [],
                    "locutions": [],
                },
                "text": {"txt": "Sample Text!"},
                "dialog": True,
                "OVA": {
                    "firstname": "",
                    "surname": "",
                    "url": "",
                    "nodes": [],
                    "edges": [],
                },
                "model_name": "",
            }
        }
    }


app = FastAPI(
    title="Argument Mining API",
    description="An API for analyzing text to identify argument components.",
    version="5",
)

# =============================================================================
# =============================================================================


# =============================================================================
# SECTION 5: The Main API Endpoint
# =============================================================================
# EXPLANATION: This is the core of API. The `@app.post("/analyze")` decorator
# tells FastAPI to create an endpoint at the URL '/analyze' that accepts POST requests.
@app.post("/analyze")
def analyze_text(request: AIFRequest):
    """
    Analyzes a piece of text to identify its argumentative components.
    """

    # Step A: Segment the incoming text into sentences using spaCy.
    input_xaif_dict = request.model_dump()
    raw_text = input_xaif_dict.get("text", {}).get("txt", "")
    doc = nlp(raw_text)

    aif_obj = aif(input_xaif_dict)
    if not raw_text:
        return aif_obj.xaif

    # Create a list to hold the structured data for each segment
    segments_data = []

    for sent in doc.sents:
        clean_text = sent.text.strip()
        if clean_text:
            # For each sentence, create a dictionary with all the required info
            segment_info = {
                "Type": "Unknown",  # We don't have this info, so we can use a placeholder, so we wont get error when the functions of the model runs
                "Text": clean_text,
            }
            segments_data.append(segment_info)

    # Convert the list of dictionaries into the rich DataFrame
    segments_df = pd.DataFrame(segments_data)
    if segments_df.empty:
        return aif_obj.xaif

    predictions = []

    # Step B: Choose the correct feature engineering path based on the user's request.
    if request.model_name == "classic_svm":
        if not classic_svm_model or not tfidf_vectorizer:
            raise HTTPException(
                status_code=500,
                detail="Classic SVM model is not available on the server.",
            )

        # TODO: Replicate FULL feature engineering pipeline here for the classic SVM.
        synt_feat = extract_syntactic_features(segments_df)
        lexi_feat = extract_lexicon_features(segments_df)
        stru_feat = extract_structural_features(segments_df)
        ngram_features = extract_ngram_features(segments_df["Text"])

        final_features = hstack(
            [
                ngram_features,
                csr_matrix(lexi_feat),
                csr_matrix(stru_feat),
                csr_matrix(synt_feat),
            ]
        )

        predictions = classic_svm_model.predict(final_features)

    elif request.model_name == "bert_svm":
        #temporary solution
        # Text = str(segments_df["Text"]) 
        # print(Text,type(Text))
        # Text = Text.strip()
        # Text = Text.lower()
        # Text = re.sub(r'[^\w\s]', '', Text)
        
        # print(Text)
        # segments_df["Text"] = Text
        # print(type(segments_df["Text"]))
        
        # print(segments_df["Text"])
        # Automated feature engineering using the BERT helper function.
        embeddings = generate_bert_embeddings(
            segments_df["Text"], tokenizer, model, device
        )
        predictions = hybrid_svm_model.predict(embeddings)
        # print(segments_df.iterrows())
        # print("BERT+SVM Predictions:", predictions)


    elif request.model_name == "roberta":
        if not roberta_model or not roberta_tokenizer:
            raise HTTPException(
                status_code=500,
                detail="RoBERTa model is not available on the server.",
            )

        # 1. Convert segmented sentences DataFrame into a Hugging Face Dataset
        inference_dataset = Dataset.from_pandas(segments_df)

        # 2. Define the tokenization function as lambda
        # 3. Tokenize the dataset and remove the original text column
        tokenized_dataset = inference_dataset.map(
            lambda examples: tokenizer(
                examples["Text"], padding="max_length", truncation=True
            ),
            batched=True,
        )
        tokenized_dataset = tokenized_dataset.remove_columns(["Text"])

        training_args = TrainingArguments(
            output_dir="/tmp/trainer_runs",  # Use the writable /tmp directory
            report_to="none",  # Disable logging to avoid other writes
        )

        # 4. Instantiate a Trainer with just the model
        # The Trainer will automatically use the GPU if available
        trainer = Trainer(
            model=roberta_model,
            args=training_args,
        )

        # 5. Use the Trainer to predict
        prediction_output = trainer.predict(tokenized_dataset)

        # 6. Get the final predictions by finding the class with the highest score
        predictions = np.argmax(prediction_output.predictions, axis=1)
    else:
        raise HTTPException(
            status_code=500,
            detail="Hybrid SVM model is not available on the server.",
        )

    # Iterate over the rows of the DataFrame using .iterrows()
    # This gives the index (i) and the content of the row (row)
    for i, row in segments_df.iterrows():
        if predictions[i] == 1:
            # Get the text from the 'Text' column of the current row
            segment_text = row["Text"]

            # Create the XAIFNode
            aif_obj.add_component("proposition", Lnode_ID=0, proposition=segment_text)

    aif_obj.xaif.pop("model_name")
    return aif_obj.xaif


# =============================================================================
# =============================================================================


# =============================================================================
# SECTION 6: Root Endpoint
# =============================================================================
# EXPLANATION: A simple endpoint at the base URL to confirm the API is running.
@app.get("/")
def read_root():
    return {"message": "Welcome to the Argument Mining API! Go to /docs to use it."}


# =============================================================================
# =============================================================================
