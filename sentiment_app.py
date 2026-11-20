"""
Tweet Sentiment Analyzer - desktop UI (Tkinter, no extra installs needed)
Loads the saved model (best_model_GRU.keras) and classifies any text you type
as POSITIVE or NEGATIVE with a confidence score.

Run:  python sentiment_app.py
"""

import os
import re
import glob
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

# Quiet TensorFlow logs (must be set before TensorFlow is imported)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs_compare", "best_model_GRU.keras")

GREEN = "#1b8a3c"
RED = "#c62828"
GREY = "#555555"

# ----------------------------------------------------------------------
# Same cleaning used during training (must match the training script!)
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


def find_model_path():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    found = glob.glob(os.path.join(BASE_DIR, "outputs_compare", "best_model_*.keras"))
    return found[0] if found else None


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.tf = None
        self.q = queue.Queue()

        root.title("Tweet Sentiment Analyzer")
        root.geometry("720x640")
        root.minsize(640, 560)

        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 10), foreground=GREY)
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=8)

        main = ttk.Frame(root, padding=16)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Tweet Sentiment Analyzer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(main, text="Type or paste a tweet / sentence and click Analyze "
                             "(Ctrl+Enter).", style="Sub.TLabel").pack(anchor="w", pady=(0, 10))

        # Input box
        self.input = tk.Text(main, height=6, wrap="word", font=("Segoe UI", 12),
                             relief="solid", borderwidth=1, padx=8, pady=8)
        self.input.pack(fill="x")
        self.input.focus_set()
        self.input.bind("<Control-Return>", lambda e: (self.analyze(), "break")[1])

        # Buttons
        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=10)
        self.analyze_btn = ttk.Button(btns, text="Analyze", style="Big.TButton",
                                      command=self.analyze, state="disabled")
        self.analyze_btn.pack(side="left")
        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left", padx=8)

        # Result card
        card = ttk.LabelFrame(main, text="Result", padding=14)
        card.pack(fill="x", pady=(4, 8))
        self.result_lbl = tk.Label(card, text="-", font=("Segoe UI", 28, "bold"), fg=GREY)
        self.result_lbl.pack()
        self.conf_lbl = tk.Label(card, text="", font=("Segoe UI", 11), fg=GREY)
        self.conf_lbl.pack(pady=(2, 6))

        bar_row = ttk.Frame(card)
        bar_row.pack(fill="x")
        ttk.Label(bar_row, text="Negative", foreground=RED).pack(side="left")
        ttk.Label(bar_row, text="Positive", foreground=GREEN).pack(side="right")
        self.bar = ttk.Progressbar(card, maximum=100, mode="determinate")
        self.bar.pack(fill="x", pady=4)
        self.prob_lbl = tk.Label(card, text="", font=("Segoe UI", 9), fg=GREY)
        self.prob_lbl.pack()

        # History
        hist_frame = ttk.LabelFrame(main, text="History (this session)", padding=6)
        hist_frame.pack(fill="both", expand=True)
        cols = ("time", "text", "sentiment", "confidence")
        self.tree = ttk.Treeview(hist_frame, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (70, 360, 90, 90)):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor="w" if c == "text" else "center")
        self.tree.tag_configure("pos", foreground=GREEN)
        self.tree.tag_configure("neg", foreground=RED)
        sb = ttk.Scrollbar(hist_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Status bar
        self.status = tk.StringVar(value="Loading model... (first start takes a few seconds)")
        ttk.Label(root, textvariable=self.status, relief="sunken", anchor="w",
                  padding=(8, 3)).pack(fill="x", side="bottom")

        # Load the model in the background so the window opens immediately
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(100, self._poll)

    # ---------------- background work ----------------
    def _load_model(self):
        try:
            path = find_model_path()
            if path is None:
                raise FileNotFoundError(
                    f"Model not found.\nExpected at:\n{MODEL_PATH}")
            import tensorflow as tf
            from tensorflow import keras
            self.tf = tf
            model = keras.models.load_model(path, compile=False)
            # warm-up call so the first real prediction is fast
            model.predict(tf.constant(["warm up"]), verbose=0)
            self.model = model
            self.q.put(("loaded", os.path.basename(path)))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", str(e)))

    def _predict(self, original, cleaned):
        try:
            score = float(self.model.predict(self.tf.constant([cleaned]), verbose=0)[0][0])
            self.q.put(("result", original, score))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", str(e)))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "loaded":
                    self.status.set(f"Model ready: {msg[1]}")
                    self.analyze_btn.config(state="normal")
                elif kind == "result":
                    self._show_result(msg[1], msg[2])
                elif kind == "error":
                    self.status.set("Error")
                    self.analyze_btn.config(state="normal" if self.model else "disabled")
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ---------------- UI actions ----------------
    def analyze(self):
        if self.model is None:
            return
        text = self.input.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty input", "Please type some text first.")
            return
        cleaned = clean_text(text)
        if not cleaned:
            messagebox.showinfo("No words found",
                                "The text has no readable words after cleaning "
                                "(only links, mentions, numbers or symbols).")
            return
        self.status.set("Analyzing...")
        self.analyze_btn.config(state="disabled")
        threading.Thread(target=self._predict, args=(text, cleaned), daemon=True).start()

    def _show_result(self, text, score):
        positive = score >= 0.5
        label = "POSITIVE" if positive else "NEGATIVE"
        color = GREEN if positive else RED
        confidence = score if positive else 1 - score

        self.result_lbl.config(text=label, fg=color)
        self.conf_lbl.config(text=f"Confidence: {confidence * 100:.1f}%", fg=color)
        self.bar["value"] = score * 100
        self.prob_lbl.config(text=f"P(positive) = {score:.3f}   |   P(negative) = {1 - score:.3f}")

        short = text.replace("\n", " ")
        short = short if len(short) <= 70 else short[:67] + "..."
        self.tree.insert("", 0,
                         values=(datetime.now().strftime("%H:%M:%S"), short, label,
                                 f"{confidence * 100:.1f}%"),
                         tags=("pos" if positive else "neg",))
        self.status.set("Done")
        self.analyze_btn.config(state="normal")

    def clear(self):
        self.input.delete("1.0", "end")
        self.result_lbl.config(text="-", fg=GREY)
        self.conf_lbl.config(text="")
        self.prob_lbl.config(text="")
        self.bar["value"] = 0
        self.input.focus_set()


if __name__ == "__main__":
    root = tk.Tk()
    SentimentApp(root)
    root.mainloop()
