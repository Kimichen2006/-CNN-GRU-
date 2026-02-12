import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# ======================
# 參數設定
# ======================
class Config:
    """集中管理所有超參數"""
    CSV_PATH = "C:/Users/user/Desktop/量化/tw_stock_data/csv/all_stocks.csv"
    WINDOW_SIZE = 20
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-5  # L2 正則化
    PATIENCE = 10  # Early stopping
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_SAVE_PATH = "tw_stock_parameter.pt"
    FEATURE_COLS = ["open", "high", "low", "close", "volume", "factor"]
    
    # 資料分割比例
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    
    # 隨機種子（可重現性）
    RANDOM_SEED = 42

# 設定隨機種子
def set_seed(seed: int = 42):
    """確保實驗可重現"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(Config.RANDOM_SEED)

# ======================
# 資料讀取與前處理
# ======================
def load_and_preprocess_data(csv_path: str, feature_cols: list) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    載入並預處理股票資料
    
    Returns:
        X: 特徵序列 (samples, window_size, features)
        y: 標籤 (samples,)
        df: 原始資料框
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # 資料驗證
    if df.empty:
        raise ValueError("CSV file is empty")
    
    if not all(col in df.columns for col in feature_cols + ["date"]):
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
    
    # 日期處理
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 移除缺失值
    df = df.dropna(subset=feature_cols)
    print(f"Data shape after cleaning: {df.shape}")
    
    # 特徵工程：增加技術指標
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """增加常用技術指標作為特徵"""
    # 收益率
    df["returns"] = df["close"].pct_change()
    
    # 移動平均
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    
    # 價格動量
    df["momentum"] = df["close"] - df["close"].shift(5)
    
    # 波動率
    df["volatility"] = df["returns"].rolling(window=10).std()
    
    # RSI (簡化版)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # 移除初始缺失值
    df = df.dropna()
    
    return df

def create_sequences(df: pd.DataFrame, feature_cols: list, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    建立時序資料（滑動視窗）
    
    改進：使用更好的標籤定義（考慮漲跌幅度）
    """
    # 更新特徵欄位（包含技術指標）
    extended_features = feature_cols + ["returns", "ma5", "ma10", "momentum", "volatility", "rsi"]
    extended_features = [col for col in extended_features if col in df.columns]
    
    data = df[extended_features].values
    
    # 標準化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    
    for i in range(window_size, len(data_scaled) - 1):
        X.append(data_scaled[i - window_size:i])
        
        # 計算下一根K棒的漲跌幅
        current_price = df["close"].iloc[i]
        next_price = df["close"].iloc[i + 1]
        price_change_pct = (next_price - current_price) / current_price
        
        # 標籤：漲跌（可以考慮設定閾值，例如只有漲跌幅>0.5%才算）
        y.append(1 if price_change_pct > 0 else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Sequences created - X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution - Up: {np.sum(y)}, Down: {len(y) - np.sum(y)}")
    
    return X, y, scaler, extended_features

# ======================
# Dataset
# ======================
class StockDataset(Dataset):
    """改進的Dataset類別"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# ======================
# 改進的 CNN + GRU Model
# ======================
class ImprovedCNN_GRU(nn.Module):
    """
    改進版本：
    1. 多層CNN提取局部特徵
    2. 雙向GRU捕捉時序依賴
    3. Dropout防止過擬合
    4. Batch Normalization加速訓練
    """
    def __init__(self, input_features: int, dropout: float = 0.3):
        super().__init__()

        # 多層卷積
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 雙向GRU
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # 全連接層
        self.fc1 = nn.Linear(128 * 2, 64)  # *2 因為是雙向
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        x = x.permute(0, 2, 1)  # (batch, features, time)
        
        # CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        
        # GRU layer
        _, h = self.gru(x)
        
        # 合併雙向隱藏狀態
        h = torch.cat([h[-2], h[-1]], dim=1)
        
        # Fully connected layers
        x = self.relu(self.fc1(h))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out

# ======================
# Early Stopping
# ======================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

# ======================
# 訓練與評估函數
# ======================
def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: str) -> float:
    """訓練一個epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        
        # Gradient clipping防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str) -> Tuple[float, float]:
    """評估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
            
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def detailed_evaluation(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    """詳細評估（混淆矩陣、分類報告等）"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            
            preds = model(X_batch)
            probs = torch.softmax(preds, dim=1)
            predicted = torch.argmax(preds, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 計算指標
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Down", "Up"], output_dict=True)
    auc = roc_auc_score(all_labels, all_probs)
    
    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "auc": auc,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs
    }

def plot_training_history(train_losses: list, val_losses: list, val_accs: list):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")

# ======================
# Main Training Pipeline
# ======================
def main():
    config = Config()
    
    print(f"Using device: {config.DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 載入資料
    df = load_and_preprocess_data(config.CSV_PATH, config.FEATURE_COLS)
    
    # 建立序列
    X, y, scaler, feature_names = create_sequences(df, config.FEATURE_COLS, config.WINDOW_SIZE)
    
    # 分割資料：Train / Val / Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SIZE / (1 - config.TEST_SIZE), shuffle=False
    )
    
    print(f"\nData split:")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 建立DataLoader
    train_loader = DataLoader(
        StockDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        StockDataset(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = DataLoader(
        StockDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # 建立模型
    model = ImprovedCNN_GRU(input_features=X.shape[2]).to(config.DEVICE)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 損失函數（可以考慮類別權重處理不平衡）
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor([len(y_train) / (2 * c) for c in class_counts], dtype=torch.float32).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    
    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    
    # 訓練循環
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        # 訓練
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # 驗證
        val_loss, val_acc = evaluate(model, val_loader, criterion, config.DEVICE)
        
        # 記錄
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 學習率調整
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler": scaler,
                "window_size": config.WINDOW_SIZE,
                "features": feature_names,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, config.MODEL_SAVE_PATH)
            print(f"  → Best model saved (Val Loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # 繪製訓練歷史
    plot_training_history(train_losses, val_losses, val_accs)
    
    # 載入最佳模型進行測試
    print("\n" + "="*50)
    print("Loading best model for final evaluation...")
    print("="*50)
    
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 測試集評估
    test_loss, test_acc = evaluate(model, test_loader, criterion, config.DEVICE)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 詳細評估
    print("\n" + "="*50)
    print("Detailed Evaluation on Test Set")
    print("="*50)
    
    eval_results = detailed_evaluation(model, test_loader, config.DEVICE)
    
    print("\nConfusion Matrix:")
    print(eval_results["confusion_matrix"])
    
    print("\nClassification Report:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 60)
    for label, metrics in eval_results["classification_report"].items():
        if label in ["Down", "Up"]:
            print(f"{label:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {metrics['support']:<12.0f}")
    
    print(f"\nAUC-ROC Score: {eval_results['auc']:.4f}")
    
    # 保存評估結果
    with open('evaluation_results.json', 'w') as f:
        # 將numpy類型轉換為Python原生類型
        results_to_save = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "auc": float(eval_results["auc"]),
            "confusion_matrix": eval_results["confusion_matrix"].tolist(),
            "classification_report": eval_results["classification_report"]
        }
        json.dump(results_to_save, f, indent=2)
    
    print("\nEvaluation results saved to 'evaluation_results.json'")
    print("\n✓ Training completed successfully!")

if __name__ == "__main__":
    main()