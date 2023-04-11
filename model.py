import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SentimentDataset(Dataset):
    """
    Subclass of torch.utils.data.Dataset, designed for sentiment analysis tasks.
    Tokenizes and prepares the input text data for use with a pre-trained transformer model.

    Attributes:
        data (pandas.DataFrame): Dataframe containing the input text data and corresponding labels.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object compatible with the pre-trained transformer model.
        max_len (int): Maximum length for the tokenized text; longer texts will be truncated.
    """

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset given its index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing the tokenized input_ids, attention_mask, and label for the requested item.
        """
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['stars'] - 1
        tokens = self.tokenizer.encode_plus(text, add_special_tokens=True,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            truncation=True,
                                            return_attention_mask=True,
                                            return_tensors='pt',
                                            )
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long),
        }


class BertGruSentimentClassifier(nn.Module):
    """
    Sentiment analysis model combining BERT and GRU layers.
    Inherits from the PyTorch Module class and is designed to classify the sentiment of a text.

    Args:
        n_classes (int): Number of output classes for sentiment classification.
    """

    def __init__(self, n_classes):
        """
        Initializes the BertGruSentimentClassifier model with the given number of output classes.

        Args:
            n_classes (int): Number of output classes for sentiment classification.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.gru = nn.GRU(input_size=self.bert.config.hidden_size,
                          hidden_size=256,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=True,
                          dropout=0.2,
                          )
        self.out = nn.Linear(512, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass of the BertGruSentimentClassifier model.

        Args:
            input_ids (torch.Tensor): Tokenized input text data, as a tensor.
            attention_mask (torch.Tensor): Attention mask for the input text data, as a tensor.

        Returns:
            out (torch.Tensor): Output logits for the sentiment classification.
        """
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state
        gru_output, _ = self.gru(last_hidden_state)
        out = self.out(gru_output[:, -1])
        return out


def prepare_input(text, tokenizer, max_len):
    """
    Prepares the input for the model by tokenizing the text, applying padding and truncation,
    and converting the output to tensors.

    Args:
        text (str): Text to tokenize.
        tokenizer (func): Function to tokenize the text.
        max_len (int): Maximum length of the text.

    Returns:
        input_ids (torch.Tensor): Input ids tensor.
        attention_mask (torch.Tensor): Attention mask tensor.
    """
    tokens = tokenizer.encode_plus(text, add_special_tokens=True,
                                   max_length=max_len,
                                   padding="max_length",
                                   truncation=True,
                                   return_attention_mask=True,
                                   return_tensors="pt",
                                   )
    input_ids = tokens["input_ids"].squeeze().to(DEVICE)
    attention_mask = tokens["attention_mask"].squeeze().to(DEVICE)
    return input_ids, attention_mask
