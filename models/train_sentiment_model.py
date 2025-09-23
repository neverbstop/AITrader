import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer # We'll use a pre-trained tokenizer for simplicity and best practices
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from models.transformer_model import SentimentTransformer # Import our custom model
import os

# Define hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 10  # Increased epochs for better training on the dummy data
LEARNING_RATE = 2e-5

class FinancialNewsDataset(Dataset):
    """
    A custom PyTorch Dataset for financial news sentiment analysis.
    This class handles tokenization and prepares the data for the model.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # Tokenize the text using a pre-trained tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the validation dataset.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            output = model(input_ids, attention_mask)
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    return accuracy, report

def train_model():
    """
    Main function to load data, train the custom transformer model,
    and save the trained model weights.
    """
    # Step 1: Data Preparation
    # For this example, we'll use a dummy dataset. In a real-world
    # scenario, you would use a large, labeled dataset of financial
    # news. You would need to create a CSV file with 'text' and 'label' columns.
    data = {'text': [
        "Apple's new iPhone launch is expected to drive record sales.",
        "A major bank reported a huge quarterly loss, causing panic.",
        "The market is showing a neutral response to the new regulations.",
        "Merger talks between two tech giants have been very positive.",
        "Massive layoffs at the company have sent the stock plunging.",
        "Analysts are divided on the future performance of the stock.",
        "A successful R&D breakthrough has significantly boosted stock prices.",
        "New government regulations could negatively impact the tech sector.",
        "Company's strong earnings beat expectations, leading to a rally.",
        "Geopolitical tensions are causing market instability and fear."
    ], 'label': [2, 0, 1, 2, 0, 1, 2, 0, 2, 0]} # 0: Negative, 1: Neutral, 2: Positive
    df = pd.DataFrame(data)

    # Split data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42) # Increased validation size for a better split

    # Load a pre-trained tokenizer. We use this because it's a critical and complex
    # component that's difficult to build from scratch. The transformer architecture
    # is what we are building, not the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets and dataloaders
    train_dataset = FinancialNewsDataset(
        texts=train_df.text.to_list(),
        labels=train_df.label.to_list(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)

    val_dataset = FinancialNewsDataset(
        texts=val_df.text.to_list(),
        labels=val_df.label.to_list(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_dataloader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE)

    # Step 2: Initialize our custom model
    # The number of classes is 3 (Negative, Neutral, Positive)
    vocab_size = tokenizer.vocab_size
    num_classes = 3
    model = SentimentTransformer(
        vocab_size=vocab_size,
        d_model=768,       # Dimension of model (matches BERT base)
        num_heads=12,      # Number of attention heads (matches BERT base)
        d_ff=3072,         # Dimension of feed-forward network (matches BERT base)
        num_layers=2,      # We'll use a small number of layers for this example
        dropout=0.1,
        num_classes=num_classes
    )

    # Step 3: Define loss function and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Step 4: Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for data in train_dataloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            optimizer.zero_grad()
            
            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate after each epoch
        accuracy, report = evaluate_model(model, val_dataloader, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {total_loss/len(train_dataloader):.4f}, Validation Accuracy: {accuracy:.4f}")
        model.train()  # Set model back to training mode

    # Step 5: Save the trained model and tokenizer
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sentiment_transformer.pth")
    torch.save(model.state_dict(), model_path)
    
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    train_model()
