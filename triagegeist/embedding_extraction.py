"""Text-embedding pipelines for chief-complaint features.

Two backends:

* **Bio_ClinicalBERT** -- Alsentzer et al. (*Clinical NLP Workshop, NAACL*
  2019), pretrained on MIMIC-III clinical notes.  Fine-tuned on the ESI
  acuity classification task with a custom ``TriageClassifier`` (BERT
  encoder + mean pooling + dropout + linear head).  After training, the
  classification head is bypassed to extract 768-d embeddings.
* **MiniLM** -- ``all-MiniLM-L6-v2`` via :mod:`sentence_transformers`,
  used frozen as a baseline.  Encodes in 10 000-row chunks with
  per-chunk ``.npy`` caching for crash resilience.

Both pipelines emit fixed-dimension embeddings that are row-aligned
with the tabular feature matrix and later PCA-reduced per-fold inside
:func:`triagegeist.data_processing.preprocess_fold`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLINICALBERT_ID = "emilyalsentzer/Bio_ClinicalBERT"
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"


# ── Hyperparameters ─────────────────────────────────────────────────────────


@dataclass
class FineTuneConfig:
    """Hyperparameters for the ClinicalBERT fine-tuning loop.

    Defaults match the Azure training run on MIMIC-IV-ED triage data.
    """

    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 16
    max_length: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    seed: int = 42


# ── MiniLM (frozen baseline) ───────────────────────────────────────────────


def encode_minilm(
    texts: list[str],
    batch_size: int = 64,
    chunk_size: int = 10_000,
    cache_dir: str | Path | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Produce 384-d sentence embeddings from a frozen MiniLM model.

    Encodes in ``chunk_size``-row chunks (default 10 000) to keep memory
    bounded on large datasets.  When ``cache_dir`` is given, each chunk
    is saved as a ``.npy`` shard so encoding can resume after a crash.

    Args:
        texts: Raw chief-complaint strings.
        batch_size: Batch size passed to ``SentenceTransformer.encode``.
        chunk_size: Number of rows per encoding chunk.
        cache_dir: Optional directory for per-chunk ``.npy`` caches.
        device: PyTorch device string (defaults to CUDA if available).

    Returns:
        Array of shape ``(len(texts), 384)`` with float32 embeddings.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM_ID, device=device)

    n_chunks = (len(texts) + chunk_size - 1) // chunk_size
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
    else:
        cache_path = None

    all_parts: list[np.ndarray] = []
    t0 = time.time()

    for i in range(n_chunks):
        # Try loading cached chunk
        if cache_path is not None:
            chunk_file = cache_path / f"minilm_chunk_{i:04d}.npy"
            if chunk_file.exists():
                part = np.load(chunk_file)
                all_parts.append(part)
                continue

        batch = texts[i * chunk_size : (i + 1) * chunk_size]
        part = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        if cache_path is not None:
            np.save(chunk_file, part)

        elapsed = time.time() - t0
        if n_chunks > 1:
            eta = elapsed / (i + 1) * (n_chunks - i - 1)
            print(
                f"  MiniLM chunk {i + 1}/{n_chunks} — "
                f"{part.shape[0]} rows  ({elapsed:.0f}s elapsed, "
                f"ETA ~{eta:.0f}s)"
            )

        all_parts.append(part)

    return np.vstack(all_parts)


# ── ClinicalBERT classifier + mean pooling ─────────────────────────────────


def _build_triage_classifier(
    model_id: str = CLINICALBERT_ID,
    num_classes: int = 5,
    dropout: float = 0.1,
):
    """Construct a ``TriageClassifier`` (BERT + mean-pool + linear head).

    The model is defined inline to keep the public package dependency-light
    (torch / transformers are only imported when this function is called).

    Returns:
        A ``torch.nn.Module`` with a ``forward(input_ids, attention_mask,
        return_embeddings=False)`` signature.  When ``return_embeddings``
        is True the classification head is bypassed and the mean-pooled
        768-d vector is returned directly.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class TriageClassifier(nn.Module):
        """Transformer encoder + mean pooling + classification head.

        Architecture::

            input_ids / attention_mask
                  │
            BERT encoder  (768-d hidden states)
                  │
            mean_pool over non-padding tokens
                  │
            Dropout(0.1)  ← skipped when return_embeddings=True
                  │
            Linear(768, 5)
        """

        def __init__(
            self,
            model_name: str,
            n_classes: int = 5,
            drop_rate: float = 0.1,
        ):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(drop_rate)
            self.classifier = nn.Linear(hidden_size, n_classes)

        # ── pooling ──────────────────────────────────────────────────────

        @staticmethod
        def mean_pool(
            last_hidden_state: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Weighted mean over non-padding tokens."""
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-8)
            return summed / counts

        # ── forward ──────────────────────────────────────────────────────

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            return_embeddings: bool = False,
        ) -> torch.Tensor:
            outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)

            if return_embeddings:
                return pooled  # 768-d, no head

            dropped = self.dropout(pooled)
            logits = self.classifier(dropped)
            return logits

    return TriageClassifier(model_id, n_classes=num_classes, drop_rate=dropout)


# ── Dataset helper ──────────────────────────────────────────────────────────


def _make_triage_dataset(texts, labels, tokenizer, max_length):
    """Create a simple map-style Dataset for triage texts."""
    import torch
    from torch.utils.data import Dataset

    class _TriageDataset(Dataset):
        def __init__(self, txt, lbl, tok, ml):
            self.texts = txt
            self.labels = lbl
            self.tokenizer = tok
            self.max_length = ml

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    return _TriageDataset(texts, labels, tokenizer, max_length)


# ── Training helpers ────────────────────────────────────────────────────────


def _train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    """Run one training epoch with optional FP16 mixed precision."""
    import torch
    import torch.nn as nn

    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"

    # Import AMP utilities (handle torch version differences)
    if use_amp:
        try:
            from torch.amp import autocast

            def cuda_autocast():
                return autocast(device_type="cuda", dtype=torch.float16)
        except ImportError:
            from torch.cuda.amp import autocast as _autocast

            def cuda_autocast():
                return _autocast(dtype=torch.float16)

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with cuda_autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(
                f"    batch {batch_idx}/{len(loader)}  "
                f"loss={loss.item():.4f}"
            )

    return total_loss / len(loader)


def _evaluate(model, loader, device):
    """Evaluate model on a DataLoader; returns (loss, macro_f1, preds, labels)."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"

    if use_amp:
        try:
            from torch.amp import autocast

            def cuda_autocast():
                return autocast(device_type="cuda", dtype=torch.float16)
        except ImportError:
            from torch.cuda.amp import autocast as _autocast

            def cuda_autocast():
                return _autocast(dtype=torch.float16)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if use_amp:
                with cuda_autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    macro_f1 = f1_score(all_labels_arr, all_preds_arr, average="macro")
    avg_loss = total_loss / len(loader)

    return avg_loss, macro_f1, all_preds_arr, all_labels_arr


def _extract_embeddings(model, loader, device) -> np.ndarray:
    """Extract mean-pooled 768-d embeddings (bypasses the classification head)."""
    import torch

    model.eval()
    all_embs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            embs = model(input_ids, attention_mask, return_embeddings=True)
            all_embs.append(embs.cpu().float().numpy())
    return np.vstack(all_embs)


# ── Public API: fine-tuning ─────────────────────────────────────────────────


def finetune_clinicalbert(
    texts: list[str],
    labels: np.ndarray,
    config: FineTuneConfig | None = None,
    output_dir: str | Path = "clinicalbert-finetuned",
    device: str | None = None,
    model_id: str = CLINICALBERT_ID,
) -> str:
    """Fine-tune Bio_ClinicalBERT on the ESI acuity classification task.

    Uses a custom ``TriageClassifier`` (BERT encoder + mean pooling +
    dropout(0.1) + Linear(768, 5)) trained with a manual PyTorch loop:

    * **Optimizer**: AdamW, lr=2e-5, weight_decay=0.01
    * **Scheduler**: linear warmup over 10 % of total steps
    * **Mixed precision**: ``GradScaler`` + ``autocast`` on CUDA
    * **Gradient clipping**: ``clip_grad_norm_(1.0)``
    * **Data split**: 80 / 10 / 10 train / val / test, stratified
    * **Best model**: checkpoint with highest validation macro F1

    After fine-tuning, call :func:`encode_clinicalbert` with
    ``model_dir=output_dir`` to extract 768-d embeddings from the
    fine-tuned checkpoint (classification head is bypassed via
    ``return_embeddings=True``).

    Args:
        texts: Raw chief-complaint strings.
        labels: Integer ESI acuity labels (1--5).
        config: Training hyperparameters; uses defaults if None.
        output_dir: Directory for the saved checkpoint.
        device: PyTorch device string (defaults to CUDA if available).
        model_id: HuggingFace model identifier for the BERT encoder.

    Returns:
        Path to the saved checkpoint directory (as a string).
    """
    import torch
    from sklearn.model_selection import train_test_split
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    cfg = config or FineTuneConfig()
    device_obj = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Remap labels to 0-indexed (CrossEntropyLoss expects 0..C-1) ──
    labels_0 = labels - 1 if labels.min() >= 1 else labels

    # ── 80/10/10 stratified split ────────────────────────────────────
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels_0, test_size=0.2, stratify=labels_0,
        random_state=cfg.seed,
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels,
        random_state=cfg.seed,
    )
    train_labels = np.asarray(train_labels)
    val_labels = np.asarray(val_labels)
    test_labels = np.asarray(test_labels)

    print(
        f"  Train: {len(train_texts):,}  Val: {len(val_texts):,}  "
        f"Test: {len(test_texts):,}"
    )

    # ── Tokenizer + datasets ─────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_ds = _make_triage_dataset(
        train_texts, train_labels, tokenizer, cfg.max_length,
    )
    val_ds = _make_triage_dataset(
        val_texts, val_labels, tokenizer, cfg.max_length,
    )
    test_ds = _make_triage_dataset(
        test_texts, test_labels, tokenizer, cfg.max_length,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, num_workers=2,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, num_workers=2,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = _build_triage_classifier(
        model_id, num_classes=5, dropout=cfg.dropout,
    ).to(device_obj)

    # ── Optimizer + scheduler ────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
    )

    # GradScaler for FP16 on CUDA
    if device_obj.type == "cuda":
        try:
            from torch.amp import GradScaler

            scaler = GradScaler()
        except ImportError:
            from torch.cuda.amp import GradScaler

            scaler = GradScaler()
    else:
        scaler = None

    print(
        f"  {cfg.num_epochs} epochs, lr={cfg.learning_rate}, "
        f"batch={cfg.batch_size}, warmup={warmup_steps}/{total_steps} steps"
    )

    # ── Training loop (best-val-F1 checkpointing) ────────────────────
    best_val_f1 = 0.0
    best_epoch = 0
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        t_epoch = time.time()
        print(f"  Epoch {epoch}/{cfg.num_epochs}")

        train_loss = _train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device_obj,
        )
        val_loss, val_f1, _, _ = _evaluate(model, val_loader, device_obj)

        elapsed = time.time() - t_epoch
        print(
            f"    train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_f1_macro={val_f1:.4f}  ({elapsed:.0f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_macro": val_f1,
            "elapsed_seconds": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), output_path / "model.pt")
            tokenizer.save_pretrained(output_path / "tokenizer")
            with open(output_path / "config.json", "w") as f:
                json.dump(
                    {
                        "base_model": model_id,
                        "num_classes": 5,
                        "max_length": cfg.max_length,
                        "dropout": cfg.dropout,
                        "best_epoch": epoch,
                        "best_val_f1": best_val_f1,
                    },
                    f,
                    indent=2,
                )
            print(f"    -> saved best model (F1={val_f1:.4f})")

    total_time = time.time() - t_start
    print(
        f"  Training complete in {total_time:.0f}s  "
        f"(best epoch {best_epoch}, val_f1={best_val_f1:.4f})"
    )

    # ── Reload best checkpoint and evaluate on held-out test set ──────
    model.load_state_dict(
        torch.load(output_path / "model.pt", map_location=device_obj)
    )
    test_loss, test_f1, test_preds, test_true = _evaluate(
        model, test_loader, device_obj,
    )
    print(f"  Test set: loss={test_loss:.4f}  macro_f1={test_f1:.4f}")

    # ── Save training results ────────────────────────────────────────
    results = {
        "base_model": model_id,
        "device": str(device_obj),
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "epochs": cfg.num_epochs,
        "lr": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "max_length": cfg.max_length,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_f1_macro": float(test_f1),
        "training_history": history,
        "total_training_seconds": total_time,
    }
    with open(output_path / "finetune_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return str(output_path)


# ── Public API: embedding extraction ────────────────────────────────────────


def encode_clinicalbert(
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 128,
    device: str | None = None,
    model_id: str = CLINICALBERT_ID,
    model_dir: str | Path | None = None,
) -> np.ndarray:
    """Produce 768-d mean-pooled embeddings from Bio_ClinicalBERT.

    Accepts a fine-tuned checkpoint in two ways (for backward
    compatibility with the Azure pipeline):

    1. ``model_dir="/path/to/checkpoint"`` -- explicit checkpoint path.
    2. ``model_id="/path/to/checkpoint"`` -- if ``model_id`` is a local
       directory containing ``model.pt``, it is treated as a checkpoint.

    In both cases the ``TriageClassifier`` is loaded from the checkpoint
    and embeddings are extracted with ``return_embeddings=True``
    (bypassing the classification head).  Otherwise the frozen
    pretrained HuggingFace model is used with mean pooling.

    Args:
        texts: Raw chief-complaint strings.
        batch_size: Inference batch size.
        max_length: Maximum token length for the tokenizer.
        device: PyTorch device string (defaults to CUDA if available).
        model_id: HuggingFace model identifier, *or* a local path to a
            fine-tuned checkpoint directory containing ``model.pt``.
        model_dir: Optional explicit path to a fine-tuned checkpoint.
            Takes precedence over ``model_id`` when both are given.

    Returns:
        Array of shape ``(len(texts), 768)`` with float32 embeddings.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    device_obj = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Resolve checkpoint path ───────────────────────────────────────
    # Support both model_dir= and model_id= pointing to a checkpoint
    checkpoint_path: Path | None = None
    if model_dir is not None:
        checkpoint_path = Path(model_dir)
    elif Path(model_id).is_dir() and (Path(model_id) / "model.pt").exists():
        checkpoint_path = Path(model_id)

    use_finetuned = (
        checkpoint_path is not None
        and (checkpoint_path / "model.pt").exists()
    )

    # ── Load model ────────────────────────────────────────────────────
    if use_finetuned:
        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                ckpt_cfg = json.load(f)
            base_model = ckpt_cfg.get("base_model", CLINICALBERT_ID)
            num_classes = ckpt_cfg.get("num_classes", 5)
            dropout = ckpt_cfg.get("dropout", 0.1)
        else:
            base_model = CLINICALBERT_ID
            num_classes = 5
            dropout = 0.1

        model = _build_triage_classifier(
            base_model, num_classes=num_classes, dropout=dropout,
        )
        model.load_state_dict(
            torch.load(
                checkpoint_path / "model.pt", map_location=device_obj,
            )
        )
        model = model.to(device_obj)

        tok_dir = checkpoint_path / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(
            str(tok_dir) if tok_dir.exists() else base_model
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = _build_triage_classifier(model_id, num_classes=5)
        model = model.to(device_obj)

    # ── Build dataset + extract ───────────────────────────────────────
    dummy_labels = np.zeros(len(texts), dtype=np.int64)
    ds = _make_triage_dataset(texts, dummy_labels, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2)

    return _extract_embeddings(model, loader, device_obj)


def extract_all_embeddings(
    texts: list[str],
    model_dir: str | Path,
    output_path: str | Path,
    batch_size: int = 32,
    max_length: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """Extract embeddings for ALL rows and save as ``.npy``.

    This mirrors the full-dataset extraction step that ran after
    fine-tuning on Azure: reload the best checkpoint, run every
    chief-complaint through the encoder with ``return_embeddings=True``,
    and persist the result.

    Args:
        texts: All chief-complaint strings (full dataset).
        model_dir: Path to the fine-tuned checkpoint directory.
        output_path: Where to save the ``.npy`` file.
        batch_size: Inference batch size.
        max_length: Maximum token length.
        device: PyTorch device string.

    Returns:
        Array of shape ``(len(texts), 768)`` with float32 embeddings.
    """
    embeddings = encode_clinicalbert(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        model_dir=model_dir,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)
    print(
        f"  Saved embeddings: {output_file}  "
        f"shape={embeddings.shape}  ({embeddings.nbytes / 1e6:.0f} MB)"
    )
    return embeddings
