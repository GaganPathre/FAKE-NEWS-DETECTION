'''Implementation of 3HAN'''

nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
nltk.download('stopwords')
nltk.download('tokenizers/punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class TextPreprocessor:
    def __init__(self, max_words=10000, max_sent_length=50, max_doc_length=20):
        self.max_words = max_words
        self.max_sent_length = max_sent_length
        self.max_doc_length = max_doc_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())

        return text

    def build_vocabulary(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()

        for text in texts:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                words = [word for word in words if word not in self.stop_words]
                word_counts.update(words)
        most_common = word_counts.most_common(self.max_words - 2)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(most_common):
            self.word_to_idx[word] = i + 2
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def text_to_sequences(self, texts):
        sequences = []

        for text in texts:
            doc_sequence = []
            sentences = sent_tokenize(text)

            for sentence in sentences[:self.max_doc_length]:
                words = word_tokenize(sentence)
                words = [word for word in words if word not in self.stop_words]

                sent_sequence = []
                for word in words[:self.max_sent_length]:
                    sent_sequence.append(self.word_to_idx.get(word, 1))  # 1 for <UNK>
                while len(sent_sequence) < self.max_sent_length:
                    sent_sequence.append(0)
                doc_sequence.append(sent_sequence)
            while len(doc_sequence) < self.max_doc_length:
                doc_sequence.append([0] * self.max_sent_length)

            sequences.append(doc_sequence)

        return np.array(sequences)
    def save_preprocessor(self, filepath):
        state = {
            'max_words': self.max_words,
            'max_sent_length': self.max_sent_length,
            'max_doc_length': self.max_doc_length,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(input_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        e = self.tanh(self.W(x))
        e = self.V(e)
        e = e.squeeze(2)

        alpha = self.softmax(e)
        alpha = alpha.unsqueeze(2)

        # Weighted sum
        output = torch.sum(alpha * x, dim=1)

        return output, alpha

class ThreeHAN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, attention_dim=64, num_classes=2):
        super(ThreeHAN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Word-level BiLSTM
        self.word_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Word-level attention
        self.word_attention = AttentionLayer(2 * hidden_dim, attention_dim)

        # Sentence-level BiLSTM
        self.sent_lstm = nn.LSTM(2 * hidden_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Sentence-level attention
        self.sent_attention = AttentionLayer(2 * hidden_dim, attention_dim)

        # Document-level BiLSTM
        self.doc_lstm = nn.LSTM(2 * hidden_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Document-level attention
        self.doc_attention = AttentionLayer(2 * hidden_dim, attention_dim)

        # Classification layer
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size, max_doc_length, max_sent_length = x.shape

        # Reshape for word-level processing
        x = x.view(batch_size * max_doc_length, max_sent_length)

        # Word embeddings
        embedded = self.embedding(x)

        # Word-level BiLSTM
        word_output, _ = self.word_lstm(embedded)

        # Word-level attention
        sent_repr, word_att = self.word_attention(word_output)

        # Reshape for sentence-level processing
        sent_repr = sent_repr.view(batch_size, max_doc_length, -1)

        # Sentence-level BiLSTM
        sent_output, _ = self.sent_lstm(sent_repr)

        # Sentence-level attention
        doc_repr, sent_att = self.sent_attention(sent_output)

        # Document-level BiLSTM
        doc_repr = doc_repr.unsqueeze(1)
        doc_output, _ = self.doc_lstm(doc_repr)

        # Document-level attention
        final_repr, doc_att = self.doc_attention(doc_output)

        # Classification
        final_repr = self.dropout(final_repr)
        logits = self.classifier(final_repr)

        return logits

class ISOTDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def load_isot_dataset(path):
    # Shuffle the dataset
    df = pd.read_csv(path)
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')

    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    accuracy = correct / total
    return total_loss / len(test_loader), accuracy, predictions, true_labels

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading Dataset...")
    path = "/content/MERGED_97.csv"

    df = load_isot_dataset(path)
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Fake news: {sum(df['label'] == 0)}, Real news: {sum(df['label'] == 1)}")

    # Preprocess text
    print("Preprocessing text...")
    preprocessor = TextPreprocessor(max_words=10000, max_sent_length=50, max_doc_length=20)
    import joblib
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("Preprocessor saved!")
    # Clean text
    df['text'] = df['text'].apply(preprocessor.clean_text)

    # Build vocabulary
    preprocessor.build_vocabulary(df['text'].values)

    # Convert to sequences
    sequences = preprocessor.text_to_sequences(df['text'].values)
    labels = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train
    )

    print(f"Training set: {len(X_train)}")
    print(f"Validation set: {len(X_val)}")
    print(f"Test set: {len(X_test)}")

    # Create datasets and data loaders
    train_dataset = ISOTDataset(X_train, y_train)
    val_dataset = ISOTDataset(X_val, y_val)
    test_dataset = ISOTDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    vocab_size = len(preprocessor.word_to_idx)
    model = ThreeHAN(vocab_size=vocab_size, embed_dim=100, hidden_dim=128, attention_dim=64, num_classes=2)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 6
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)

    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc, predictions, true_labels = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Fake', 'Real']))

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Save model
    torch.save(model.state_dict(), '3HAN.pth')
    print("Model saved as '3HAN'")

if __name__ == "__main__":
    main()

'''TESTING 3HAN'''

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    def __init__(self, max_words=10000, max_sent_length=50, max_doc_length=20):
        self.max_words = max_words
        self.max_sent_length = max_sent_length
        self.max_doc_length = max_doc_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())

        return text

    def text_to_sequences(self, texts):
        """Convert texts to sequences of word indices"""
        if isinstance(texts, str):
            texts = [texts]

        sequences = []

        for text in texts:
            doc_sequence = []
            sentences = sent_tokenize(text)

            for sentence in sentences[:self.max_doc_length]:
                words = word_tokenize(sentence)
                words = [word for word in words if word not in self.stop_words]

                sent_sequence = []
                for word in words[:self.max_sent_length]:
                    sent_sequence.append(self.word_to_idx.get(word, 1))  # 1 for <UNK>

                # Pad sentence to max_sent_length
                while len(sent_sequence) < self.max_sent_length:
                    sent_sequence.append(0)  # 0 for <PAD>

                doc_sequence.append(sent_sequence)

            # Pad document to max_doc_length
            while len(doc_sequence) < self.max_doc_length:
                doc_sequence.append([0] * self.max_sent_length)

            sequences.append(doc_sequence)

        return np.array(sequences)

    def save_preprocessor(self, filepath):
        """Save preprocessor state"""
        state = {
            'max_words': self.max_words,
            'max_sent_length': self.max_sent_length,
            'max_doc_length': self.max_doc_length,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_preprocessor(self, filepath):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.max_words = state['max_words']
        self.max_sent_length = state['max_sent_length']
        self.max_doc_length = state['max_doc_length']
        self.word_to_idx = state['word_to_idx']
        self.idx_to_word = state['idx_to_word']

class ThreeHAN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, attention_dim=64, num_classes=2):
        super(ThreeHAN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Word-level BiLSTM
        self.word_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Word-level attention
        self.word_attention = AttentionLayer(2 * hidden_dim, attention_dim)

        # Sentence-level BiLSTM
        self.sent_lstm = nn.LSTM(2 * hidden_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Sentence-level attention
        self.sent_attention = AttentionLayer(2 * hidden_dim, attention_dim)

        # Document-level BiLSTM
        self.doc_lstm = nn.LSTM(2 * hidden_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Document-level attention
        self.doc_attention = AttentionLayer(2 * hidden_dim, attention_dim)

        # Classification layer
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, max_doc_length, max_sent_length)
        batch_size, max_doc_length, max_sent_length = x.shape

        # Reshape for word-level processing
        x = x.view(batch_size * max_doc_length, max_sent_length)

        # Word embeddings
        embedded = self.embedding(x)  # (batch_size * max_doc_length, max_sent_length, embed_dim)

        # Word-level BiLSTM
        word_output, _ = self.word_lstm(embedded)

        # Word-level attention
        sent_repr, word_att = self.word_attention(word_output)

        # Reshape for sentence-level processing
        sent_repr = sent_repr.view(batch_size, max_doc_length, -1)

        # Sentence-level BiLSTM
        sent_output, _ = self.sent_lstm(sent_repr)

        # Sentence-level attention
        doc_repr, sent_att = self.sent_attention(sent_output)

        # Document-level BiLSTM
        doc_repr = doc_repr.unsqueeze(1)  # Add sequence dimension
        doc_output, _ = self.doc_lstm(doc_repr)

        # Document-level attention
        final_repr, doc_att = self.doc_attention(doc_output)

        # Classification
        final_repr = self.dropout(final_repr)
        logits = self.classifier(final_repr)

        return logits

class NewsPredictor:
    def __init__(self, model_path, preprocessor_path, device=None):

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load preprocessor
        self.preprocessor = TextPreprocessor()
        self.preprocessor.load_preprocessor(preprocessor_path)

        # Initialize and load model
        vocab_size = len(self.preprocessor.word_to_idx)
        self.model = ThreeHAN(vocab_size=vocab_size, embed_dim=100, hidden_dim=128, attention_dim=64, num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Class labels
        self.labels = {0: 'Fake', 1: 'Real'}

        print(f"Model loaded successfully on {self.device}")
        print(f"Vocabulary size: {vocab_size}")

    def predict(self, text):
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        sequence = self.preprocessor.text_to_sequences([cleaned_text])

        # Convert to tensor
        input_tensor = torch.tensor(sequence, dtype=torch.long).to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        result = {
            'prediction': self.labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'fake': probabilities[0][0].item(),
                'real': probabilities[0][1].item()
            }
        }

        return result

    def predict_batch(self, texts):
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        sequences = self.preprocessor.text_to_sequences(cleaned_texts)
        input_tensor = torch.tensor(sequences, dtype=torch.long).to(self.device)

        # Get predictions
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

        results = []
        for i in range(len(texts)):
            pred_class = predicted_classes[i].item()
            confidence = probabilities[i][pred_class].item()

            result = {
                'prediction': self.labels[pred_class],
                'confidence': confidence,
                'probabilities': {
                    'fake': probabilities[i][0].item(),
                    'real': probabilities[i][1].item()
                }
            }
            results.append(result)
        return results

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(input_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        e = self.tanh(self.W(x))
        e = self.V(e)
        e = e.squeeze(2)

        alpha = self.softmax(e)
        alpha = alpha.unsqueeze(2)

        # Weighted sum
        output = torch.sum(alpha * x, dim=1)

        return output, alpha

def main():
    predictor = NewsPredictor(
        model_path="/content/drive/MyDrive/Model/3HAN.pth",
        preprocessor_path="/content/drive/MyDrive/Model/preprocessor.pkl"
    )

    # Example news articles for testing
    fake = [
    "US Air Force suspends SpaceX project amid national security concerns after internal audit",
    "President Trump escalates feud with Elon Musk following remarks on electric vehicle policies",
    "Tesla's global sales decline for the second consecutive quarter, raising concerns among investors",
    "Israel constructs barrier around West Bank town following surge in regional violence",
    "Bedouin communities in Israel face renewed threats from nearby settler expansions, human rights groups report",
    "Eyewitness describes terrifying moment furniture and vehicles were swept away by Texas flash floods",
    "Tropical Storm Chantal makes landfall in South Carolina, bringing heavy rainfall and 70 mph winds",
    "Birkenstock files global complaint targeting counterfeit sandal sales across India and Southeast Asia",
    "Trump announces potential trade talks with China over controversial TikTok ownership legislation",
    "Investors grow anxious as Trump administration nears deadline for potential tariff expansion on imports",
    "Nissan considers transferring electric vehicle production to Foxconn in bid to save Japanese plant",
    "Kyiv reports largest drone assault since the beginning of the Ukraine-Russia conflict, military says",
    "Zelenskyy says recent conversation with Trump was 'constructive' on future aid and rebuilding plans",
    "Hollywood horror films featuring vampires and zombies continue to dominate summer box office charts",
    "OPEC+ agrees to accelerate oil output hikes starting in August to stabilize global crude prices",
    "Wall Street remains steady despite growing concerns over renewed US-China trade tensions",
    "Former Janus analyst sentenced in UK court for participating in years-long insider trading operation",
    "Mortgage rates fall for fifth straight week, offering relief to prospective homebuyers nationwide",
    "Novak Djokovic’s daughter steals the spotlight during his Wimbledon championship celebration",
    "Ten-year-old North Texas girl confirmed dead after devastating flash floods ravaged local community",
    "Filmmaker Ken Burns responds to proposed public broadcasting budget cuts, calls move 'foolhardy'",
    "U.S. deports dozens of undocumented migrants from Asia and Latin America to South Sudan detention",
    "Death toll rises to at least 66 in Texas following severe flash floods, rescue operations underway",
    "Israeli strike damages Tehran prison facility, killing multiple civilians and raising international alarms",
    "French intelligence accuses Chinese diplomats of undercutting Rafale fighter jet deals with misinformation",
    "Pope Francis to visit remote Italian monastery to restore 'body and spirit' amid papal health concerns",
    "Ken Burns premieres documentary on the American Revolution to critical acclaim at PBS headquarters",
    "Fox News reports on mid-air confrontation involving fighter jets during NATO training exercise",
    "Passenger on commercial flight confronts traveler for aggressively reclining seat on long-haul journey",
    "Nissan issues urgent recall for 480,000 vehicles citing engine stalling and fire hazard risks",
    "President Trump signs historic tax and spending bill at annual White House summer picnic",
    "Democratic leadership sees tax reform bill as opportunity to regroup ahead of upcoming elections",
    "Rapper LL Cool J pulls out of July 4th Philadelphia music festival due to scheduling conflicts",
    "Michael Madsen, best known for 'Reservoir Dogs', passes away at age 67, tributes pour in",
    "Czech Film Festival opens with global film stars and a tribute to pioneering female directors",
    "Colorado man re-arrested in connection with 2005 cold case involving wife’s suspicious disappearance",
    "Government releases hurricane preparedness guide focused on safeguarding pets and livestock",
    "National Weather Service reports that flash floods remain leading cause of storm-related fatalities",
    "Drone strikes disrupt air travel across Russia and Ukraine as regional tensions escalate further",
    "Iran’s Supreme Leader makes first public appearance since missile attack on central government building",
    "Dalai Lama turns 90, spiritual leaders around the world send congratulatory messages and blessings",
    "Elon Musk announces formation of new political movement following dispute with Trump campaign",
    "Julian McMahon, actor from 'Nip/Tuck', dies at 56 after long battle with chronic illness",
    "City councilor in northern Mexico gunned down during charity basketball game in front of children",
    "EMTs diagnose rare neurological condition in man initially thought to be drunk at sports bar",
    "Escaped zoo lion injures Turkish farmer before being tranquilized and returned by animal control",
    "Trump warns of tariff surges in July 2025 during fiery campaign speech in industrial Michigan",
    "Nevada health agency announces division split to improve services in public health and child welfare",
    "Marines deployed to Florida ICE facility amid rising tension and staff shortages post-migration surge",
    "Pope urges faithful to care for environment during World Day of Prayer for Creation in Vatican address",
    "Congress debates controversial legislation on AI surveillance and biometric data use in public spaces"
    ]

    # Single prediction
    print("\n" + "="*50)
    print("SINGLE PREDICTION EXAMPLE")
    print("="*50)

    result = predictor.predict(fake[0])
    print(f"Article: {fake[0][:100]}...")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: Fake={result['probabilities']['fake']:.4f}, Real={result['probabilities']['real']:.4f}")

    # Batch prediction
    print("\n" + "="*50)
    print("BATCH PREDICTION EXAMPLE")
    print("="*50)

    results = predictor.predict_batch(fake)

    fake_count = 0
    real_count = 0

    for i, (article, result) in enumerate(zip(fake, results)):
        print(f"\nArticle {i+1}: {article[:100]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: Fake={result['probabilities']['fake']:.4f}, Real={result['probabilities']['real']:.4f}")

        # Count predictions
        if result['prediction'].lower() == 'fake':
            fake_count += 1
        else:
            real_count += 1

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total Predictions: {len(results)}")
    print(f"Fake Predictions: {fake_count}")
    print(f"Real Predictions: {real_count}")

    # Interactive prediction
    print("\n" + "="*50)
    print("INTERACTIVE PREDICTION")
    print("="*50)
    print("Enter news articles to classify (type 'quit' to exit):")

    while True:
        user_input = input("\nEnter news article: ")
        if user_input.lower() == 'quit':
            break

        if user_input.strip():
            result = predictor.predict(user_input)
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: Fake={result['probabilities']['fake']:.4f}, Real={result['probabilities']['real']:.4f}")

if __name__ == "__main__":
    main()