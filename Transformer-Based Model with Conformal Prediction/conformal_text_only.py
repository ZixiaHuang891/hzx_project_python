import pandas as pd  # type: ignore
from typing import List
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import warnings

data_dir = "../data"
df = pd.read_csv(data_dir + "/train.csv")
TEST_SIZE = 0.1
EPOCHS = 1
N_MODELS = 8
# models to take for prediction
ENSEMBLE_MODELS = 1
RETRAIN = True


warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)


# ===== Load and preprocess data =====
def load_and_preprocess_data():
    train_df = pd.read_csv(data_dir + "/train.csv")
    test_df = pd.read_csv(data_dir + "/test.csv")
    sample_df = pd.read_csv(data_dir + "/sample.csv")

    # Get genre column names (from sample.csv)
    genre_columns = list(sample_df.columns)[1:]  # all columns except id

    # Order of features (from most important to least important)
    feature_order = [
        "title",
        "overview",
        "tagline",
        "original_title",
        "release_date",
        "runtime",
        "vote_average",
        "vote_count",
        "budget",
        "revenue",
        "original_language",
        "status",
        "homepage",
    ]

    # Fill NaN values in all columns we'll use
    for col in feature_order:
        if col in train_df.columns:
            # Fill text columns with empty string
            if col in ["runtime", "vote_average", "budget", "revenue"]:
                train_df[col] = (
                    train_df[col]
                    .astype(str)
                    .replace("0", "(no data)")
                    .replace("0.0", "(no data)")
                    .replace("nan", "(no data)")
                )
                test_df[col] = (
                    test_df[col]
                    .astype(str)
                    .replace("0", "(no data)")
                    .replace("0.0", "(no data)")
                    .replace("nan", "(no data)")
                )
            else:
                train_df[col] = (
                    train_df[col].astype(str).replace("nan", "(no data)")
                )
                test_df[col] = (
                    test_df[col].astype(str).replace("nan", "(no data)")
                )

    # Fill release_date NaN values
    for df in [train_df, test_df]:
        df["release_date"] = df["release_date"].fillna("")

    # Create formatted text with bullet points
    for df in [train_df, test_df]:
        text_items = []

        for i, row in df.iterrows():
            row_text = []

            # Add each feature as a bullet point in the specified order
            for col in feature_order:
                if col in df.columns:
                    value = row[col]
                    # Format column name with spaces and title case
                    col_name = " ".join(
                        word.capitalize() for word in col.split("_")
                    )

                    row_text.append(f"* {col_name}: {value}")

            # Join all bullet points with newlines
            # text_items.append('\n'.join(row_text))
            # separate with SEP
            text_items.append(tokenizer.sep_token.join(row_text))

        df["text"] = text_items

    # Get labels (20 genre columns as a matrix)

    ensemble_test_df = train_df.sample(
        n=int(len(train_df) * TEST_SIZE), random_state=42
    )
    train_df = train_df.drop(ensemble_test_df.index)
    # train_labels = train_df[genre_columns].values

    kf = KFold(n_splits=N_MODELS, shuffle=True, random_state=42)

    train_index, val_index = next(kf.split(train_df))
    # Split into training and validation sets
    # train_texts, val_texts, train_labels, val_labels = train_test_split(
    #     train_df["text"].values,
    #     train_labels,
    #     test_size=TEST_SIZE,
    #     # random_state=42
    # )
    data = []
    for train_index, val_index in kf.split(train_df):
        train_texts = train_df["text"].values[train_index]
        val_texts = train_df["text"].values[val_index]
        train_labels = train_df[genre_columns].values[train_index]
        val_labels = train_df[genre_columns].values[val_index]

        data.append(
            {
                "train_texts": train_texts,
                "val_texts": val_texts,
                "train_labels": train_labels,
                "val_labels": val_labels,
                "test_texts": test_df["text"].values,
                "test_ids": test_df["id"].values,
                "genre_columns": genre_columns,
                "ensemble_test_texts": ensemble_test_df["text"].values,
                "ensemble_test_labels": ensemble_test_df[genre_columns].values,
                "ensemble_test_ids": ensemble_test_df["id"].values,
            }
        )
    return data
    # train_texts = train_df["text"].values[train_index]
    # val_texts = train_df["text"].values[val_index]
    # train_labels = train_df[genre_columns].values[train_index]
    # val_labels = train_df[genre_columns].values[val_index]

    # return {
    #     "train_texts": train_texts,
    #     "val_texts": val_texts,
    #     "train_labels": train_labels,
    #     "val_labels": val_labels,
    #     "test_texts": test_df["text"].values,
    #     "test_ids": test_df["id"].values,
    #     "genre_columns": genre_columns,
    #     "ensemble_test_texts": ensemble_test_df["text"].values,
    #     "ensemble_test_labels": ensemble_test_df[genre_columns].values,
    #     "ensemble_test_ids": ensemble_test_df["id"].values,
    # }


# ===== DistilBert-based multi-label classification model =====
class GenreClassifier(nn.Module):
    def __init__(self, n_genres):
        super(GenreClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_genres)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ===== Dataset and DataLoader classes =====
class MovieDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer.encode_plus(  # type: ignore
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        if self.labels is not None:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(self.labels[idx], dtype=torch.float),
            }
        else:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
            }


# ===== Model training function =====
def train_model(
    data,
    num_epochs=5,
    batch_size=16,
    learning_rate=3e-5,
    log_steps=10,
    name="model",
):
    train_dataset = MovieDataset(
        texts=data["train_texts"],
        labels=data["train_labels"],
        tokenizer=tokenizer,
    )

    val_dataset = MovieDataset(
        texts=data["val_texts"], labels=data["val_labels"], tokenizer=tokenizer
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    n_genres = data["train_labels"].shape[1]
    model = GenreClassifier(n_genres=n_genres)
    model = model.to(device)

    # Use class weights if unbalanced
    pos_weight = torch.ones(n_genres).to(device)
    for i in range(n_genres):
        # Calculate positive samples ratio for each genre
        pos_count = data["train_labels"][:, i].sum()
        neg_count = len(data["train_labels"]) - pos_count
        pos_weight[i] = neg_count / (
            pos_count + 1e-5
        )  # Add small epsilon to avoid division by zero

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_data_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Binary cross entropy loss with class weights
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize lists to store metrics for visualization
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_f1_scores = []
    epoch_precision = []
    epoch_recall = []

    # For logging steps
    step_losses = []
    global_steps = []

    # Training loop
    best_f1 = 0
    best_model_state = None
    patience = 2
    patience_counter = 0
    global_step = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_losses = []

        # Track running loss for more frequent printing
        running_loss = 0.0

        for batch_idx, batch in enumerate(
            tqdm(train_data_loader, desc="Training")
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_value = loss.item()
            train_losses.append(loss_value)
            running_loss += loss_value

            # Print loss every log_steps batches
            if batch_idx % log_steps == 0:
                avg_running_loss = running_loss / (
                    batch_idx + 1 if batch_idx > 0 else 1
                )
                print(
                    f"  Step {global_step}, Batch {batch_idx}, Train Loss: {avg_running_loss:.4f}"
                )

                # Optional: store step-wise losses for more detailed plotting
                step_losses.append(avg_running_loss)
                global_steps.append(global_step)

        # Epoch-level train loss
        avg_train_loss = np.mean(train_losses)
        epoch_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_true = []

        running_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_data_loader, desc="Validation")
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                loss = loss_fn(outputs, labels)

                loss_value = loss.item()
                val_losses.append(loss_value)
                running_val_loss += loss_value

                # Print validation loss periodically
                if batch_idx % log_steps == 0:
                    avg_running_val_loss = running_val_loss / (
                        batch_idx + 1 if batch_idx > 0 else 1
                    )
                    print(
                        f"  Validation Batch {batch_idx}, Val Loss: {avg_running_val_loss:.4f}"
                    )

                preds = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)

        # Try different thresholds to find the best one
        best_threshold = 0.5
        # best_f1_score = 0

        # for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        #     binary_preds = (val_preds > threshold).astype(int)
        #     f1 = f1_score(val_true, binary_preds, average='micro', zero_division=0)
        #     if f1 > best_f1_score:
        #         best_f1_score = f1
        #         best_threshold = threshold

        # Use the best threshold
        binary_preds = (val_preds > best_threshold).astype(int)

        # Calculate evaluation metrics
        precision = precision_score(
            val_true, binary_preds, average="micro", zero_division=0
        )
        recall = recall_score(
            val_true, binary_preds, average="micro", zero_division=0
        )
        f1 = f1_score(val_true, binary_preds, average="micro", zero_division=0)

        # Store metrics for visualization
        avg_val_loss = np.mean(val_losses)
        epoch_val_losses.append(avg_val_loss)
        epoch_f1_scores.append(f1)
        epoch_precision.append(precision)
        epoch_recall.append(recall)

        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
        print(f"Best Threshold: {best_threshold}")
        print(
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
        )

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot training and validation loss
    plt.figure(figsize=(15, 10))

    # Plot epoch losses
    plt.subplot(2, 2, 1)
    epochs = range(1, len(epoch_train_losses) + 1)
    plt.plot(epochs, epoch_train_losses, "b-o", label="Training Loss")
    plt.plot(epochs, epoch_val_losses, "r-o", label="Validation Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot step-wise training loss
    plt.subplot(2, 2, 2)
    plt.plot(global_steps, step_losses, "g-", label="Step-wise Training Loss")
    plt.title("Training Loss per Step")
    plt.xlabel("Global Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot metrics
    plt.subplot(2, 2, 3)
    plt.plot(epochs, epoch_f1_scores, "g-o", label="F1 Score")
    plt.plot(epochs, epoch_precision, "y-o", label="Precision")
    plt.plot(epochs, epoch_recall, "m-o", label="Recall")
    plt.title("Performance Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # Plot loss vs metrics
    plt.subplot(2, 2, 4)
    plt.scatter(
        epoch_val_losses,
        epoch_f1_scores,
        c=range(len(epoch_val_losses)),
        cmap="viridis",
        s=100,
    )
    for i, epoch_num in enumerate(range(1, len(epoch_val_losses) + 1)):
        plt.annotate(
            f"{epoch_num}",
            (epoch_val_losses[i], epoch_f1_scores[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.colorbar(label="Epoch")
    plt.title("Validation Loss vs F1 Score")
    plt.xlabel("Validation Loss")
    plt.ylabel("F1 Score")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{name}_training_metrics_detailed.pdf", dpi=300)
    plt.show()

    # Create a table of metrics per epoch
    metrics_table = pd.DataFrame(
        {
            "Epoch": range(1, len(epoch_train_losses) + 1),
            "Train Loss": epoch_train_losses,
            "Val Loss": epoch_val_losses,
            "F1 Score": epoch_f1_scores,
            "Precision": epoch_precision,
            "Recall": epoch_recall,
        }
    )

    # Also create a step-wise loss table
    step_metrics = pd.DataFrame(
        {"Global Step": global_steps, "Training Loss": step_losses}
    )

    print("\nTraining Metrics per Epoch:")
    print(metrics_table)

    # Save metrics to CSV
    metrics_table.to_csv(f"{name}_training_metrics_per_epoch.csv", index=False)
    step_metrics.to_csv(f"{name}_training_metrics_per_step.csv", index=False)

    return model, tokenizer, best_threshold, metrics_table, step_metrics


def plot_training_metrics(metrics_path="training_metrics.csv"):
    """
    Plot training metrics from a saved CSV file.

    Parameters:
    metrics_path (str): Path to the CSV file containing training metrics
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load metrics
    metrics = pd.read_csv(metrics_path)

    plt.figure(figsize=(16, 8))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(
        metrics["Epoch"], metrics["Train Loss"], "b-o", label="Training Loss"
    )
    plt.plot(
        metrics["Epoch"], metrics["Val Loss"], "r-o", label="Validation Loss"
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot F1 Score
    plt.subplot(2, 2, 2)
    plt.plot(metrics["Epoch"], metrics["F1 Score"], "g-o", label="F1 Score")
    plt.title("F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # Plot Precision and Recall
    plt.subplot(2, 2, 3)
    plt.plot(metrics["Epoch"], metrics["Precision"], "y-o", label="Precision")
    plt.plot(metrics["Epoch"], metrics["Recall"], "m-o", label="Recall")
    plt.title("Precision and Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # Plot Loss vs F1 Score
    plt.subplot(2, 2, 4)
    plt.scatter(
        metrics["Val Loss"],
        metrics["F1 Score"],
        c=metrics["Epoch"],
        cmap="viridis",
        s=100,
    )
    for i, epoch in enumerate(metrics["Epoch"]):
        plt.annotate(
            f"Epoch {epoch}",
            (metrics["Val Loss"][i], metrics["F1 Score"][i]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.colorbar(label="Epoch")
    plt.title("Validation Loss vs F1 Score")
    plt.xlabel("Validation Loss")
    plt.ylabel("F1 Score")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics_detailed.png", dpi=300)
    plt.show()

    return metrics


def nonconformity_score_max(
    predictions: torch.Tensor, true_labels: torch.Tensor
) -> torch.Tensor:
    """
    Computes nonconformity scores for a batch of samples.
    The nonconformity score is defined as the sum of probabilities up to
    the worst-ranked true label (i.e., the true label with the lowest probability).

    Args:
        predictions (torch.Tensor): A 2D tensor (batch_size, num_classes) with predicted probabilities.
        true_labels (torch.Tensor): A 2D binary tensor indicating true labels (batch_size x num_classes).

    Returns:
        scores (torch.Tensor): Nonconformity scores for each sample.
                               Samples with no true labels will have a score of None.
    """
    batch_size = predictions.size(0)
    scores = []

    for i in range(batch_size):
        p = predictions[i]
        labels = true_labels[i]

        if labels.sum() == 0:
            # If no true labels are present, you might decide to handle this case separately.
            scores.append(None)
            continue

        # Sort the predicted probabilities in descending order.
        sorted_probs, sorted_indices = torch.sort(p, descending=True)

        # Create a tensor for ranks (0-indexed: 0 is highest probability).
        ranks = torch.arange(len(p), device=p.device)
        rank_positions = torch.empty_like(sorted_indices)
        rank_positions[sorted_indices] = ranks
        # for debugging
        labels_positions = torch.where(labels == 1)  # type: ignore
        # Get the ranks for the true labels.
        true_label_ranks = rank_positions[labels.bool()]
        # Find the worst (maximum) rank among the true labels.
        worst_rank = true_label_ranks.max()

        # Compute the nonconformity score: sum of probabilities up to worst_rank (inclusive)
        # score = sorted_probs[: worst_rank + 1].sum()
        score = sorted_probs[:worst_rank].sum()
        scores.append(score.item())

    return torch.tensor(scores)


# ===== Functions for conformal prediction =====
def compute_calibration_scores_max(model, data, tokenizer, batch_size=16):
    model.eval()

    cal_dataset = MovieDataset(
        texts=data["val_texts"], labels=data["val_labels"], tokenizer=tokenizer
    )

    cal_data_loader = DataLoader(cal_dataset, batch_size=batch_size)

    nonconformity_scores = []

    with torch.no_grad():
        for batch in tqdm(cal_data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)

            # Calculate nonconformity scores for each sample and class

            nonconf_score = nonconformity_score_max(probs, labels)
            nonconformity_scores.extend(nonconf_score.cpu().tolist())

    return np.array(nonconformity_scores)  # individual score (one per sample)


# ===== Functions for conformal prediction =====
def compute_calibration_scores(model, data, tokenizer, batch_size=16):
    model.eval()

    cal_dataset = MovieDataset(
        texts=data["val_texts"], labels=data["val_labels"], tokenizer=tokenizer
    )

    cal_data_loader = DataLoader(cal_dataset, batch_size=batch_size)

    nonconformity_scores = []

    with torch.no_grad():
        for batch in tqdm(cal_data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)

            # Calculate nonconformity scores for each sample and class
            # For label=1, score is 1-prob; for label=0, score is prob
            # just exclude when label is 0 ???
            # right now 20 prbs
            # inverts the probability when the label is 1, leaves as-is if
            # label is 0
            nonconf_score = torch.where(
                labels == 1,
                (1 - probs),
                probs,
            )
            nonconf_score = torch.where(
                labels == 1, (1 - probs), (probs)
            )  # 20 cluster probe
            nonconformity_scores.append(nonconf_score.cpu().numpy())

    nonconformity_scores = np.vstack(nonconformity_scores)

    return nonconformity_scores  # 20 cluster score


# ===== BASIC Functions for conformal prediction for submission =====
def predict_with_conformal_max(
    model,
    data,
    tokenizer,
    nonconformity_scores,
    threshold=0.5,
    alpha=0.1,
    batch_size=16,
    test_field="test_texts",
    test_ids="test_ids",
):
    # Create dataset and dataloader for test data
    test_dataset = MovieDataset(texts=data[test_field], tokenizer=tokenizer)

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Calculate quantile for conformal prediction
    q = np.quantile(nonconformity_scores, 1 - alpha, axis=0)

    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().tolist())

    # Convert the numpy array of probabilities to a tensor.
    all_probs_tensor = torch.tensor(all_probs)

    standard_preds = all_probs_tensor > threshold

    # Sort each row in descending order.
    # sorted_probs contains the sorted values and sorted_indices contains the original indices.
    sorted_probs, sorted_indices = torch.sort(
        all_probs_tensor, dim=1, descending=True
    )

    # Initialize a tensor for conformal predictions with the same shape as all_probs_tensor.
    conformal_preds = torch.zeros_like(all_probs_tensor)

    # For each sample, select labels based on cumulative sum reaching the threshold q.
    for i in range(sorted_indices.size(0)):
        # Get sorted probabilities for the i-th sample.
        sample_sorted_probs = sorted_probs[i]

        # Compute the cumulative sum over sorted probabilities.
        cum_probs = torch.cumsum(sample_sorted_probs, dim=0)

        # Find the smallest k such that the cumulative sum >= q.
        k_candidates = (cum_probs >= q).nonzero(as_tuple=True)[0]

        if k_candidates.numel() == 0:
            # If the cumulative sum never reaches q, select all indices.
            selected = sorted_indices[i]
        else:
            # We need to include indices up to and including the first index that reaches the threshold.
            k = k_candidates[0].item() + 1  # +1 because indices are 0-indexed.
            selected = sorted_indices[i, :k]

        # Set the corresponding indices in the conformal prediction tensor to 1.
        conformal_preds[i, selected] = 1

    # Create submission dataframe
    submission = pd.DataFrame({"id": data[test_ids]})
    for i, genre in enumerate(data["genre_columns"]):
        submission[genre] = conformal_preds[:, i]

    return submission, all_probs, q, standard_preds


# ===== BASIC Functions for conformal prediction for submission =====
def predict_with_conformal(
    model,
    data,
    tokenizer,
    nonconformity_scores,
    threshold=0.5,
    alpha=0.1,
    batch_size=16,
    test_field="test_texts",
    test_ids="test_ids",
):
    # Create dataset and dataloader for test data
    test_dataset = MovieDataset(texts=data[test_field], tokenizer=tokenizer)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    # Calculate quantile for conformal prediction
    q = np.quantile(nonconformity_scores, 1 - alpha, axis=0)

    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)

    # Construct conformal prediction sets
    # Include genre in set if probability >= (1-q)
    conformal_preds = (all_probs >= (1 - q)).astype(int)

    # Also create standard predictions with the best threshold
    standard_preds = (all_probs > threshold).astype(int)

    # Ensure each movie has at least one genre
    # If no genre is predicted, assign the one with highest probability
    for i in range(conformal_preds.shape[0]):
        if np.sum(conformal_preds[i]) == 0:
            # Find genre with highest probability
            max_prob_idx = np.argmax(all_probs[i])
            conformal_preds[i, max_prob_idx] = 1

    # Create submission dataframe
    submission = pd.DataFrame({"id": data[test_ids]})
    for i, genre in enumerate(data["genre_columns"]):
        submission[genre] = conformal_preds[:, i]

    return submission, all_probs, q, standard_preds


# ===== Evaluation functions based on the competition metric =====
# from sympy import N


def evaluate_conformal_prediction(
    model,
    data,
    tokenizer,
    # nonconformity_scores,
    threshold,
    alpha=0.1,
    conformal_scores_function=compute_calibration_scores,
    prediction_function=predict_with_conformal,
):
    """
    Evaluate the model using the competition metric:
    Metric = 0.5 * (Coverage + Length)

    Coverage = Indicator[Average coverage >= 0.9]
    Length = 1 - Average set size / 20

    The ideal prediction set should have coverage >= 0.9 with minimal size.
    """
    # Split the validation data for evaluation
    val_texts, eval_texts, val_labels, eval_labels = train_test_split(
        data["val_texts"], data["val_labels"], test_size=0.5, random_state=42
    )

    # Update data dictionary for calibration
    cal_data = {
        "train_texts": data["train_texts"],
        "val_texts": val_texts,
        "train_labels": data["train_labels"],
        "val_labels": val_labels,
        "test_texts": eval_texts,
        "test_ids": np.arange(len(eval_texts)),
        "genre_columns": data["genre_columns"],
        "ensemble_test_texts": data["ensemble_test_texts"],
        "ensemble_test_labels": data["ensemble_test_labels"],
        "ensemble_test_ids": data["ensemble_test_ids"],
    }

    # Calculate calibration scores on the validation set
    cal_scores = conformal_scores_function(model, cal_data, tokenizer)

    # Predict with conformal prediction on the evaluation set
    submission, _, q, standard_preds = prediction_function(
        model, cal_data, tokenizer, cal_scores, threshold, alpha
    )

    # Calculate coverage (fraction of instances where true labels are in the prediction set)
    conformal_preds = submission.iloc[:, 1:].values
    true_labels = eval_labels

    # Calculate coverage for each sample
    coverages = []
    set_sizes = []

    for i in range(len(true_labels)):
        # Check if all true genres are in the prediction set
        # Y^(i) ⊆ S^(i)
        covered = True
        for j in range(len(true_labels[i])):
            if true_labels[i][j] == 1 and conformal_preds[i][j] == 0:
                covered = False
                break

        coverages.append(covered)

        # Calculate set size |S^(i)|
        set_size = np.sum(conformal_preds[i])
        set_sizes.append(set_size)

    # Calculate average coverage
    avg_coverage = np.mean(coverages)
    # Calculate average set size
    avg_set_size = np.mean(set_sizes)

    # Calculate the final metrics
    coverage_indicator = 1 if avg_coverage >= 0.9 else 0
    length_score = 1 - (avg_set_size / 20)

    # Final metric
    final_metric = 0.5 * (coverage_indicator + length_score)

    print(f"Evaluation Results:")
    print(f"  cal score shape {cal_scores.shape}")
    # print(f"  nonconformity_scores shape {nonconformity_scores.shape}")
    print(f"  q: {q}")
    print(f"  1-q (vs softmax score): {1-q}")
    print(f"  Average Coverage: {avg_coverage:.4f}")
    print(f"  Coverage Indicator (≥0.9): {coverage_indicator}")
    print(f"  Average Set Size: {avg_set_size:.4f}")
    print(f"  Length Score: {length_score:.4f}")
    print(f"  Final Metric: {final_metric:.4f}")

    # Also evaluate standard predictions
    precision = precision_score(
        true_labels, standard_preds, average="micro", zero_division=0
    )
    recall = recall_score(
        true_labels, standard_preds, average="micro", zero_division=0
    )
    f1 = f1_score(
        true_labels, standard_preds, average="micro", zero_division=0
    )

    print(f"Standard Prediction Metrics:")
    print(
        f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )

    return final_metric, coverage_indicator, length_score, q


def vote(
    submissions: List[pd.DataFrame], trigger_happy_factor=1
) -> pd.DataFrame:
    """
    Computes the element-wise majority vote from a list of binary DataFrames.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames with binary values.

    Returns:
    pd.DataFrame: DataFrame with the majority vote for each cell.
    """
    n = len(submissions)
    # Calculate the majority threshold. For an odd number of dfs,
    # this is n//2 + 1; adjust as needed for even number cases.
    threshold = n // 2 + 1

    # Sum all dataframes element-wise. Use dfs[0] as the starting point.
    vote_sum = sum(submissions[1:], submissions[0].copy())

    # Create the majority vote DataFrame: 1 if votes >= threshold, else 0.
    vote_df = (vote_sum >= threshold * trigger_happy_factor).astype(int)

    return vote_df


# ===== Evaluation functions based on the competition metric =====
def evaluate_conformal_prediction_ensemble(
    models,
    all_data,
    tokenizer,
    # nonconformity_scores,
    threshold,
    alpha=0.1,
    conformal_scores_function=compute_calibration_scores,
    prediction_function=predict_with_conformal,
):
    """
    Evaluate the model using the competition metric:
    Metric = 0.5 * (Coverage + Length)

    Coverage = Indicator[Average coverage >= 0.9]
    Length = 1 - Average set size / 20

    The ideal prediction set should have coverage >= 0.9 with minimal size.
    """
    submissions = []
    # for i in range(N_MODELS):
    for i in range(ENSEMBLE_MODELS):
        data = all_data[i]
        model = models[i]["model"]

        # Split the validation data for evaluation
        val_texts, eval_texts, val_labels, eval_labels = train_test_split(
            data["val_texts"],
            data["val_labels"],
            test_size=0.5,
            random_state=42,
        )

        # Update data dictionary for calibration
        cal_data = {
            "train_texts": data["train_texts"],
            "val_texts": val_texts,
            "train_labels": data["train_labels"],
            "val_labels": val_labels,
            "test_texts": eval_texts,
            "test_ids": np.arange(len(eval_texts)),
            "genre_columns": data["genre_columns"],
            "ensemble_test_texts": data["ensemble_test_texts"],
            "ensemble_test_labels": data["ensemble_test_labels"],
            "ensemble_test_ids": data["ensemble_test_ids"],
        }

        # Calculate calibration scores on the validation set
        cal_scores = conformal_scores_function(model, cal_data, tokenizer)

        # Predict with conformal prediction on the evaluation set
        submission, _, q, standard_preds = prediction_function(
            model,
            cal_data,
            tokenizer,
            cal_scores,
            threshold,
            alpha=alpha,
            test_field="ensemble_test_texts",
            test_ids="ensemble_test_ids",
        )
        submissions.append(submission)

    submission = vote(submissions, trigger_happy_factor=1)

    # Calculate coverage (fraction of instances where true labels are in the prediction set)
    conformal_preds = submission.iloc[:, 1:].values
    true_labels = data["ensemble_test_labels"]

    # Calculate coverage for each sample
    coverages = []
    set_sizes = []

    for i in range(len(true_labels)):
        # Check if all true genres are in the prediction set
        # Y^(i) ⊆ S^(i)
        covered = True
        for j in range(len(true_labels[i])):
            if true_labels[i][j] == 1 and conformal_preds[i][j] == 0:
                covered = False
                break

        coverages.append(covered)

        # Calculate set size |S^(i)|
        set_size = np.sum(conformal_preds[i])
        set_sizes.append(set_size)

    # Calculate average coverage
    avg_coverage = np.mean(coverages)
    # Calculate average set size
    avg_set_size = np.mean(set_sizes)

    # Calculate the final metrics
    coverage_indicator = 1 if avg_coverage >= 0.9 else 0
    length_score = 1 - (avg_set_size / 20)

    # Final metric
    final_metric = 0.5 * (coverage_indicator + length_score)

    print(f"Evaluation Results:")
    print(f"  cal score shape {cal_scores.shape}")
    # print(f"  nonconformity_scores shape {nonconformity_scores.shape}")
    print(f"  q: {q}")
    print(f"  1-q (vs softmax score): {1-q}")
    print(f"  Average Coverage: {avg_coverage:.4f}")
    print(f"  Coverage Indicator (≥0.9): {coverage_indicator}")
    print(f"  Average Set Size: {avg_set_size:.4f}")
    print(f"  Length Score: {length_score:.4f}")
    print(f"  Final Metric: {final_metric:.4f}")

    # Also evaluate standard predictions
    precision = precision_score(
        true_labels, standard_preds, average="micro", zero_division=0
    )
    recall = recall_score(
        true_labels, standard_preds, average="micro", zero_division=0
    )
    f1 = f1_score(
        true_labels, standard_preds, average="micro", zero_division=0
    )

    print(f"Standard Prediction Metrics:")
    print(
        f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )

    return final_metric, coverage_indicator, length_score, q


def find_optimal_alpha(
    model,
    data,
    tokenizer,
    # nonconformity_scores,
    threshold,
    alpha_values=None,
    conformal_scores_function=compute_calibration_scores,
    prediction_function=predict_with_conformal,
):
    """
    Find the optimal alpha value for conformal prediction by evaluating
    different alpha values using the competition metric.
    """
    if alpha_values is None:
        # Default range of alpha values to try
        alpha_values = [
            0.01,
            0.02,
            0.025,
            0.026,
            0.027,
            0.028,
            0.029,
            0.03,
            0.031,
            0.032,
            0.033,
            0.034,
            0.035,
            0.036,
            0.037,
            0.038,
            0.039,
            0.04,
            0.05,
            0.052,
            0.055,
        ]

    results = []
    best_metric = -1
    best_alpha = None
    best_q = None

    print("Finding optimal alpha value for conformal prediction...")
    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}:")
        metric, coverage, length, q = evaluate_conformal_prediction(
            model,
            data,
            tokenizer,
            # nonconformity_scores,
            threshold,
            alpha,
            conformal_scores_function,
            prediction_function,
        )

        results.append(
            {
                "alpha": alpha,
                "metric": metric,
                "coverage": coverage,
                "length": length,
            }
        )

        if metric > best_metric:
            best_metric = metric
            best_alpha = alpha
            best_q = q

    print("\nOptimal alpha value:")
    print(f"  Alpha: {best_alpha}")
    print(f"  Metric: {best_metric:.4f}")

    return best_alpha, best_q
