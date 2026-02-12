import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import random
import numpy as np
from ditto_light.dataset import DittoDataset
from ditto_light.ditto import train
from ditto_light.ditto import DittoModel, evaluate
import time
from sklearn import metrics
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse

def train_ditto(train_txt, valid_txt, test_txt, run_name,
                lm="distilbert", max_len=256,
                batch_size=64, n_epochs=10, device=None):
    """
    Addestra Ditto su train/validation/test e salva checkpoint.

    Args:
        train_txt (str): path file train.txt (left \t right \t label)
        valid_txt (str): path file validation.txt (left \t right \t label)
        test_txt  (str): path file test.txt (left \t right \t label)
        run_name (str): nome del run / cartella checkpoint
        lm (str): modello linguistico (distilbert o roberta)
        max_len (int): lunghezza massima sequenze
        batch_size (int): batch size
        n_epochs (int): numero di epoche
        device (str, optional): "cuda" o "cpu". Se None, rileva automaticamente
    """

    # 🔹 Rileva dispositivo
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 🔹 Carica dataset
    print("[INFO] Loading datasets...")
    train_dataset = DittoDataset(train_txt, lm=lm, max_len=max_len)
    valid_dataset = DittoDataset(valid_txt, lm=lm, max_len=max_len)
    test_dataset  = DittoDataset(test_txt,  lm=lm, max_len=max_len)

    print(f"[INFO] Train size: {len(train_dataset)}")
    print(f"[INFO] Validation size: {len(valid_dataset)}")
    print(f"[INFO] Test size: {len(test_dataset)}")

    # 🔹 Set seed per riproducibilità
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # 🔹 Crea oggetto "hp" compatibile con Ditto
    hp = argparse.Namespace()
    hp.run_id = 0
    hp.batch_size = batch_size
    hp.max_len = max_len
    hp.lr = 3e-5
    hp.n_epochs = n_epochs
    hp.finetuning = True
    hp.save_model = True
    hp.logdir = "checkpoints/"
    hp.lm = lm
    hp.fp16 = False
    hp.da = None
    hp.alpha_aug = 0.8
    hp.dk = None
    hp.summarize = False
    hp.size = None
    hp.device = device
    hp.task = run_name  # 🔹 essenziale per evitare AttributeError

    # 🔹 Avvia il training
    print("[INFO] Starting training...")
    train(
        train_dataset,
        valid_dataset,
        test_dataset,   # ← passiamo il test set reale
        run_name,
        hp
    )

    print(f"[INFO] Training completed. Checkpoints saved under run_name: {run_name}")

def evaluate_ditto_model(checkpoint_path, test_txt, lm='distilbert', max_len=256, batch_size=64, device=None):
    """
    Carica il modello Ditto e valuta sul file test.txt
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # dataset test
    test_dataset = DittoDataset(test_txt, lm=lm, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=DittoDataset.pad)

    # modello
    model = DittoModel(lm=lm, device=device)
    model.to(device)

    # checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("[INFO] Model loaded successfully.")

    # evaluate
    # print("[INFO] Evaluating model...")
    # f1, best_th = evaluate(model, test_loader)
    # print(f"[RESULT] F1 Score: {f1:.4f}, Best threshold: {best_th:.2f}")
    # return f1, best_th

    # evaluate
    print("[INFO] Evaluating model...")
    f1 = evaluate(model, test_loader, threshold=0.7)
    print(f"[RESULT] F1 Score: {f1:.4f}")
    return f1
