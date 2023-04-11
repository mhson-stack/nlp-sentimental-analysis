import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from model import SentimentDataset, BertGruSentimentClassifier

MAX_LEN = 128
BATCH_SIZE = 100
N_CLASSES = 5
EPOCHS = 1
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Trains the model for one epoch and returns the average loss.

    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): DataLoader instance providing the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        scheduler (Scheduler): Learning rate scheduler.
        device (torch.device): Device to use for running the model and data.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluates the model on a dataset and returns the average loss and accuracy.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader instance providing the evaluation data.
        device (torch.device): Device to use for running the model and data.

    Returns:
        float: Average loss for the evaluation.
        float: Accuracy for the evaluation.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    progress_bar = tqdm(dataloader, desc="Evaluation", unit="batch")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        loss = nn.CrossEntropyLoss()(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)


def main(args):
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)

    train_dataset = SentimentDataset(
        train_df[['text', 'stars']], tokenizer, MAX_LEN)
    val_dataset = SentimentDataset(
        val_df[['text', 'stars']], tokenizer, MAX_LEN)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BertGruSentimentClassifier(N_CLASSES)
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(
            train_dataloader) * EPOCHS
    )

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, DEVICE)
        print(f'Train loss: {train_loss:.4f}')

        val_loss, val_acc = evaluate(model, val_dataloader, DEVICE)
        print(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')

    torch.save(model.state_dict(), 'bert_gru_sentiment_classifier.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict sentiment using a trained BERT-GRU classifier")
    parser.add_argument('--train_path', type=str, default='data/train.csv',
                        help='Path to the train dataset')
    parser.add_argument('--val_path', type=str,
                        default='data/valid.csv', help='Path to the valid dataset')

    args = parser.parse_args()
    main(args)
