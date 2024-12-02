from datasets import DatasetDict
from transformers.tokenization_utils import BatchEncoding

LABEL_NAMES = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

def align_labels_with_tokens_single(
    labels: list,
    tokenized_text: BatchEncoding,
    label_names: list = LABEL_NAMES,
):
    word_ids = tokenized_text.word_ids()  # map from token indices to word indices
    aligned_labels = []
    
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:  # tokens for cls and sep
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            aligned_labels.append(labels[word_id])
        else:
            aligned_label = labels[word_id]
            if label_names[aligned_label].startswith("B-"):
                aligned_label = label_names.index("I-" + label_names[aligned_label][2:])
            aligned_labels.append(aligned_label)
        prev_word_id = word_id
    
    return aligned_labels

def align_labels_with_tokens(
    labels: list, 
    word_ids: list, 
    label_names: list = LABEL_NAMES,
):
    aligned_labels = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            aligned_labels.append(labels[word_id])
        else:
            aligned_label = labels[word_id]
            if label_names[aligned_label].startswith("B-"):
                aligned_label = label_names.index("I-" + label_names[aligned_label][2:])
            aligned_labels.append(aligned_label)
        prev_word_id = word_id
    return aligned_labels


def prepare_dataset(
    dataset: DatasetDict, 
    tokenizer,
    max_length: int = 128,
):
    """
    Tokenizes and aligns labels for a given dataset.

    Args:
    - dataset (DatasetDict): dataset containing tokens and ner_tags.
    - tokenizer: The tokenizer to use.
    - label_names (list): List of label names.
    - max_length (int): Maximum sequence length.

    Returns:
    - processed_dataset
    """
    def tokenize_and_align_labels(examples):
        tokenized_inputs: BatchEncoding = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        
        all_labels = []
        all_tokens = []
        for i, labels in enumerate(examples["ner_tags"]):
            tokenizer_tokens = tokenized_inputs.tokens(batch_index=i)
            all_tokens.append(tokenizer_tokens)

            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels = align_labels_with_tokens(labels, word_ids)
            all_labels.append(aligned_labels)
        
        tokenized_inputs["labels"] = all_labels
        tokenized_inputs["tokenizer_tokens"] = all_tokens
        return tokenized_inputs

    processed_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    processed_dataset = processed_dataset.remove_columns(["tokens", "ner_tags"])
    processed_dataset.set_format(type="torch", columns=["tokenizer_tokens", "input_ids", "attention_mask", "labels"])

    return processed_dataset
