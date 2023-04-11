import argparse
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score
from model import prepare_input, BertGruSentimentClassifier
from train import tokenizer
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(text_list, model, tokenizer, max_len=128):
    """
    Predicts the sentiment of a list of texts using the given model and tokenizer.

    Args:
        text_list (pd.DataFrame): List of texts.
        model (nn.Module): Model to use for prediction.
        tokenizer (func): Function to tokenize the text.
        max_len (int, optional): Maximum length of the text. Defaults to 128.

    Returns:
        predictions (list): List of predictions.
    """
    predictions = []
    for text in tqdm(text_list):
        input_ids, attention_mask = prepare_input(text, tokenizer, max_len)
        with torch.no_grad():
            logits = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item() + 1
        predictions.append(prediction)
    return predictions


def score_report(y_true, y_pred):
    """
    Prints the classification report and the confusion matrix for the true and predicted labels.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
    """
    print(classification_report(y_true, y_pred))
    print('accuracy', accuracy_score(y_true, y_pred))



def main(args):
    df = pd.read_csv(args.test_path)
    model = BertGruSentimentClassifier(5)
    model.load_state_dict(torch.load(
        "bert_gru_sentiment_classifier.pth", map_location=torch.device(DEVICE)))
    model = model.to(DEVICE)
    model.eval()

    prediction = predict(df['text'], model, tokenizer)
    if 'stars' in df.columns:
        score_report(df['stars'], prediction)
    pd.concat([df['review_id'], pd.DataFrame((prediction), columns=['stars'])],
              axis=1).to_csv(f'{args.save_path}test_pred.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict sentiment using a trained BERT-GRU classifier")
    parser.add_argument('--test_path', type=str, default='data/valid.csv',
                        help='Path to the predicting dataset')
    parser.add_argument('--model_path', type=str,
                        default='bert_gru_sentiment_classifier.pth', help='Path to the trained model')
    parser.add_argument('--save_path', type=str,
                        default='data/', help='Path to the trained model')
    args = parser.parse_args()

    main(args)
