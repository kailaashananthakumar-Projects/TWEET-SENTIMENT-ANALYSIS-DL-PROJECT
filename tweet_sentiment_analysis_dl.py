"""
Tweet Sentiment Analysis (Sentiment140) - RNN vs LSTM vs BiLSTM vs GRU vs BiGRU
-------------------------------------------------------------------------------
* Same data, same vocabulary, same regularisation for every model (fair comparison)
* Overfitting controls: SpatialDropout, dropout, L2 weight decay, gradient clipping,
  EarlyStopping (restore best weights), ReduceLROnPlateau, small embedding/hidden size
* Metrics: accuracy, precision, recall, specificity, F1, macro-F1, ROC-AUC, MCC,
  log-loss, train-vs-test gap (overfit check), training time, parameter count
* Outputs (in ./outputs_compare): comparison CSV, training curves, metric bars,
  ROC curves, confusion matrices, overfitting-gap chart, best end-to-end model
"""

import os
import re
import time
import random

# Silence oneDNN info message (must be set before importing tensorflow)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
    log_loss,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ----------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------
CSV_PATH = "tweet_sentiment.csv"
SAMPLE_SIZE = 200_000        # total rows used (balanced: half per class)
MAX_TOKENS = 20_000          # vocabulary size
MAX_LEN = 40                 # tokens per tweet
EMBED_DIM = 64
UNITS = 64                   # hidden units (per direction for Bi- models)
BATCH_SIZE = 256
MAX_EPOCHS = 10              # EarlyStopping usually stops earlier
PATIENCE = 2
SEED = 42
OUTPUT_DIR = "outputs_compare"
SHOW_PLOTS = False           # True -> also open plot windows
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))


def finish_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------
# 2. Load + balance data
# ----------------------------------------------------------------------
cols = ["target", "ids", "date", "flag", "user", "text"]
df = pd.read_csv(CSV_PATH, encoding="latin-1", names=cols)
df = df[["target", "text"]]
print("Raw shape:", df.shape)
print(df["target"].value_counts())

per_class = SAMPLE_SIZE // 2
df = (
    pd.concat(
        [
            df[df["target"] == 0].sample(per_class, random_state=SEED),
            df[df["target"] == 4].sample(per_class, random_state=SEED),
        ]
    )
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)
df["label"] = df["target"].map({0: 0, 4: 1})  # 0 = negative, 1 = positive
print("Final dataset size:", len(df))

# ----------------------------------------------------------------------
# 3. Clean text
# ----------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HTML_RE = re.compile(r"&\w+;")
NON_ALPHA_RE = re.compile(r"[^a-z\s']")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = NON_ALPHA_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

# ----------------------------------------------------------------------
# 4. Split (80 / 10 / 10, stratified)
# ----------------------------------------------------------------------
X = df["clean_text"].to_numpy(dtype=object)
y = df["label"].to_numpy(dtype="float32")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ----------------------------------------------------------------------
# 5. Vectorize once (vocabulary learned on TRAIN only -> no leakage)
# ----------------------------------------------------------------------
vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN,
)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(1024))
print("Vocabulary size:", len(vectorizer.get_vocabulary()))


def to_sequences(texts, chunk=20_000):
    out = []
    for i in range(0, len(texts), chunk):
        out.append(vectorizer(tf.constant(list(texts[i:i + chunk]))).numpy())
    return np.concatenate(out).astype("int32")


Xs_train = to_sequences(X_train)
Xs_val = to_sequences(X_val)
Xs_test = to_sequences(X_test)


def make_ds(xs, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((xs, labels))
    if shuffle:
        ds = ds.shuffle(20_000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(Xs_train, y_train, shuffle=True)
train_eval_ds = make_ds(Xs_train, y_train)      # un-shuffled, for evaluation
val_ds = make_ds(Xs_val, y_val)
test_ds = make_ds(Xs_test, y_test)

# ----------------------------------------------------------------------
# 6. Model builders (identical regularisation for all)
# ----------------------------------------------------------------------
DROPOUT_IN = 0.3      # dropout on the recurrent layer inputs
DROPOUT_HEAD = 0.5    # dropout before the classifier
L2 = 1e-5             # weight decay on dense layers


def rnn_layer(kind):
    if kind == "RNN":
        return layers.SimpleRNN(UNITS, dropout=DROPOUT_IN)
    if kind == "LSTM":
        return layers.LSTM(UNITS, dropout=DROPOUT_IN)
    if kind == "GRU":
        return layers.GRU(UNITS, dropout=DROPOUT_IN)
    if kind == "BiLSTM":
        return layers.Bidirectional(layers.LSTM(UNITS, dropout=DROPOUT_IN))
    if kind == "BiGRU":
        return layers.Bidirectional(layers.GRU(UNITS, dropout=DROPOUT_IN))
    raise ValueError(kind)


def build_model(kind):
    keras.utils.set_random_seed(SEED)   # same init seed for every model
    inputs = keras.Input(shape=(MAX_LEN,), dtype="int32")
    x = layers.Embedding(MAX_TOKENS, EMBED_DIM, mask_zero=True)(inputs)
    x = layers.SpatialDropout1D(0.3)(x)
    x = rnn_layer(kind)(x)
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Dropout(DROPOUT_HEAD)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name=kind)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),  # clipnorm: stable RNN
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


MODEL_NAMES = ["RNN", "LSTM", "BiLSTM", "GRU", "BiGRU"]

# ----------------------------------------------------------------------
# 7. Metrics helper
# ----------------------------------------------------------------------
def compute_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds),
        "Recall (Sensitivity)": recall_score(y_true, preds),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, preds),
        "Macro F1": f1_score(y_true, preds, average="macro"),
        "ROC-AUC": roc_auc_score(y_true, probs),
        "MCC": matthews_corrcoef(y_true, preds),
        "Log Loss": log_loss(y_true, probs),
    }


# ----------------------------------------------------------------------
# 8. Train + evaluate every model
# ----------------------------------------------------------------------
y_test_int = y_test.astype(int)
results, histories, probs_all, cms, trained = [], {}, {}, {}, {}

for name in MODEL_NAMES:
    print("\n" + "=" * 70)
    print(f"Training {name}")
    print("=" * 70)

    model = build_model(name)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5
        ),
    ]

    t0 = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - t0
    histories[name] = hist.history

    # Evaluate (weights restored to the best epoch by EarlyStopping)
    train_loss, train_acc = model.evaluate(train_eval_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    probs = model.predict(Xs_test, batch_size=512, verbose=0).ravel()
    probs_all[name] = probs

    m = compute_metrics(y_test_int, probs)
    gap = train_acc - m["Accuracy"]
    verdict = "OK" if gap < 0.03 else ("Mild" if gap < 0.06 else "OVERFIT")

    print(f"\n--- {name}: test classification report ---")
    print(classification_report(
        y_test_int, (probs >= 0.5).astype(int),
        target_names=["Negative", "Positive"], digits=4))

    cms[name] = confusion_matrix(y_test_int, (probs >= 0.5).astype(int))
    results.append({
        "Model": name,
        **m,
        "Train Acc": train_acc,
        "Val Acc": val_acc,
        "Train-Test Gap": gap,
        "Overfit?": verdict,
        "Epochs Run": len(hist.history["loss"]),
        "Params": model.count_params(),
        "Train Time (s)": round(train_time, 1),
    })
    trained[name] = model

# ----------------------------------------------------------------------
# 9. Comparison table
# ----------------------------------------------------------------------
res = pd.DataFrame(results).set_index("Model")
res_sorted = res.sort_values("F1", ascending=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
print("\n" + "=" * 70)
print("FINAL COMPARISON (sorted by F1 on the test set)")
print("=" * 70)
print(res_sorted.round(4).to_string())

res_sorted.round(4).to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
best_name = res_sorted.index[0]
print(f"\nBest model by F1: {best_name}")

# ----------------------------------------------------------------------
# 10. Plots
# ----------------------------------------------------------------------
palette = dict(zip(MODEL_NAMES, sns.color_palette("tab10", len(MODEL_NAMES))))

# 10a. Training curves (solid = validation, dashed = train)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name in MODEL_NAMES:
    h = histories[name]
    c = palette[name]
    axes[0].plot(h["loss"], "--", color=c, alpha=0.6)
    axes[0].plot(h["val_loss"], "-", color=c, label=name)
    axes[1].plot(h["accuracy"], "--", color=c, alpha=0.6)
    axes[1].plot(h["val_accuracy"], "-", color=c, label=name)
axes[0].set_title("Loss  (dashed = train, solid = validation)")
axes[1].set_title("Accuracy  (dashed = train, solid = validation)")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
finish_plot("training_curves.png")

# 10b. Metric bars
metric_cols = ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity",
               "F1", "ROC-AUC"]
plot_df = res[metric_cols].reset_index().melt(
    id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(13, 5))
ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model",
                 hue_order=MODEL_NAMES, palette=palette)
lo = max(0.0, plot_df["Score"].min() - 0.03)
ax.set_ylim(lo, min(1.0, plot_df["Score"].max() + 0.02))
ax.set_title("Test-set metrics by model")
ax.grid(axis="y", alpha=0.3)
finish_plot("metric_comparison.png")

# 10c. ROC curves
plt.figure(figsize=(6.5, 6))
for name in MODEL_NAMES:
    fpr, tpr, _ = roc_curve(y_test_int, probs_all[name])
    plt.plot(fpr, tpr, color=palette[name],
             label=f"{name} (AUC = {res.loc[name, 'ROC-AUC']:.4f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves (test set)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
finish_plot("roc_curves.png")

# 10d. Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, name in zip(axes.ravel(), MODEL_NAMES):
    sns.heatmap(cms[name], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
for ax in axes.ravel()[len(MODEL_NAMES):]:
    ax.axis("off")
finish_plot("confusion_matrices.png")

# 10e. Overfitting check: train accuracy vs test accuracy
x = np.arange(len(MODEL_NAMES))
w = 0.27
plt.figure(figsize=(9, 5))
plt.bar(x - w, res.loc[MODEL_NAMES, "Train Acc"], w, label="Train")
plt.bar(x, res.loc[MODEL_NAMES, "Val Acc"], w, label="Validation")
plt.bar(x + w, res.loc[MODEL_NAMES, "Accuracy"], w, label="Test")
plt.xticks(x, MODEL_NAMES)
lo = min(res["Train Acc"].min(), res["Accuracy"].min()) - 0.03
plt.ylim(lo, min(1.0, res["Train Acc"].max() + 0.02))
plt.ylabel("Accuracy")
plt.title("Overfitting check: train vs validation vs test accuracy")
plt.legend()
plt.grid(axis="y", alpha=0.3)
finish_plot("overfitting_check.png")

# 10f. Time vs accuracy
plt.figure(figsize=(7, 5))
for name in MODEL_NAMES:
    plt.scatter(res.loc[name, "Train Time (s)"], res.loc[name, "Accuracy"],
                s=140, color=palette[name], label=name)
plt.xlabel("Training time (s)")
plt.ylabel("Test accuracy")
plt.title("Accuracy vs training time")
plt.legend()
plt.grid(alpha=0.3)
finish_plot("time_vs_accuracy.png")

# ----------------------------------------------------------------------
# 11. Save best model as an end-to-end text model + sample predictions
# ----------------------------------------------------------------------
text_in = keras.Input(shape=(), dtype=tf.string, name="text")
final_model = keras.Model(text_in, trained[best_name](vectorizer(text_in)),
                          name=f"{best_name}_text_model")
final_path = os.path.join(OUTPUT_DIR, f"best_model_{best_name}.keras")
final_model.save(final_path)
print(f"Saved best model ({best_name}) to {final_path}")

samples = [
    "I absolutely love this phone, the camera is amazing!",
    "Worst service ever. I am so disappointed and angry.",
    "Just had a great day at the beach with friends :)",
    "My flight got cancelled again. This is awful.",
]
cleaned = tf.constant([clean_text(s) for s in samples])
scores = final_model.predict(cleaned, verbose=0).ravel()
print(f"\nSample predictions ({best_name}):")
for s, p in zip(samples, scores):
    print(f"  [{'Positive' if p >= 0.5 else 'Negative':8s} {p:.3f}] {s}")

print(f"\nAll results and plots are in: {os.path.abspath(OUTPUT_DIR)}")