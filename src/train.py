"""
# Fase 1: Preparação e Serialização do Modelo (Pipeline Offline - src/train.py)
Script responsável por:
1. Baixar/Carregar a base oficial MNIST (`mnist_784`, 70.000 imagens 28x28) via `fetch_openml` do Scikit-Learn.
2. Verificar e tratar eventuais valores nulos (imputação preventiva), converter os rótulos categóricos ('0'..'9')
   para inteiros (0..9) e realizar a divisão estratificada em conjuntos de Treino, Validação e Teste
   (exatamente antes de qualquer ajuste para evitar data leakage), escalonando os pixels para [0.0, 1.0] (/ 255.0).
3. Treinar o modelo campeão XGBoost Classifier com os hiperparâmetros otimizados no Mini-Projeto 2
   (`n_estimators=200`, `max_depth=7`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.7`).
4. Avaliar o modelo no conjunto de validação e no conjunto de teste final (Acurácia, Precisão, Recall, F1-Score, ROC-AUC).
5. Serializar o artefato treinado em `models/xgboost_mnist.joblib` e salvar o relatório detalhado de métricas
   em `models/metrics_report.json`, além de exportar imagens reais de exemplo para `data/samples/`.
"""

import json
import sys
import time
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path mesmo se executado diretamente
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# Diretórios relativos ao projeto (Fase 5.1: proibido caminhos absolutos fixos da máquina)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OPENML_CACHE_DIR = DATA_DIR / "openml_cache"


def train_and_serialize_pipeline() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FASE 1: PREPARAÇÃO, TREINAMENTO E SERIALIZAÇÃO DO MODELO XGBOOST (MNIST)")
    print("=" * 80)

    # 1. Carregamento do dataset MNIST (mnist_784)
    print("[1/5] Carregando dataset MNIST (mnist_784) via OpenML / Scikit-Learn...")
    t0 = time.perf_counter()
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
        data_home=str(OPENML_CACHE_DIR)
    )
    X_raw, y_raw = mnist.data, mnist.target
    print(f"      Dataset carregado em {time.perf_counter() - t0:.2f}s | Shape X: {X_raw.shape}, Shape y: {y_raw.shape}")

    # 2. Tratamento de nulos, codificação de rótulos e Divisão Estrita (Treino / Validação / Teste)
    print("[2/5] Verificando integridade dos dados e aplicando divisão Treino / Validação / Teste...")
    null_count = int(np.isnan(X_raw).sum())
    if null_count > 0:
        print(f"      Imputando {null_count} valores ausentes com 0.0 (fundo preto)...")
        X_raw = np.nan_to_num(X_raw, nan=0.0)
    else:
        print("      Nenhum valor nulo encontrado (0 NaNs).")

    # Codificação dos rótulos alvo para inteiros de 0 a 9
    y_int = y_raw.astype(int)

    # Divisão estratificada: 70% Treino (49.000), 10% Validação (7.000) e 20% Teste (14.000)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_int, test_size=0.20, random_state=42, stratify=y_int
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp  # 0.125 * 0.80 = 0.10 (10%)
    )

    # Escalonamento / Normalização para [0.0, 1.0] dividindo por 255.0 após o split
    X_train = (X_train / 255.0).astype(np.float32)
    X_val = (X_val / 255.0).astype(np.float32)
    X_test = (X_test / 255.0).astype(np.float32)

    print(
        f"      Divisão concluída: Treino={X_train.shape[0]} | "
        f"Validação={X_val.shape[0]} | Teste={X_test.shape[0]}"
    )

    # Exporta amostras reais do conjunto de teste (0 a 9) em fundo preto padrão MNIST e fundo branco invertido
    print("      Exportando imagens de amostra (0 a 9) para data/samples/...")
    for digit in range(10):
        idx = int(np.where(y_test == digit)[0][0])
        sample_28x28 = (X_test[idx].reshape(28, 28) * 255.0).astype(np.uint8)
        # Salva versão padrão MNIST (fundo preto, letra branca)
        Image.fromarray(sample_28x28, mode="L").save(SAMPLES_DIR / f"mnist_digito_{digit}_fundo_preto.png")
        # Salva versão papel (fundo branco, letra escura) para demonstrar a inversão automática do predict.py
        sample_inverted = 255 - sample_28x28
        Image.fromarray(sample_inverted, mode="L").save(SAMPLES_DIR / f"mnist_digito_{digit}_fundo_claro.png")

    # 3. Treinamento do Modelo Campeão XGBoost Classifier
    print("[3/5] Treinando o modelo XGBoost Classifier com hiperparâmetros otimizados...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.7,
        objective="multi:softprob",
        num_class=10,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    t_train_start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time_sec = time.perf_counter() - t_train_start
    print(f"      Treinamento concluído em {train_time_sec:.2f} segundos.")

    # 4. Avaliação de Métricas nos conjuntos de Validação e Teste
    print("[4/5] Calculando métricas de desempenho (Acurácia, Precisão, Recall, F1-Score, ROC-AUC)...")
    t_pred_start = time.perf_counter()
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    pred_time_sec = time.perf_counter() - t_pred_start

    val_acc = float(accuracy_score(y_val, y_val_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_precision = float(precision_score(y_test, y_test_pred, average="weighted"))
    test_recall = float(recall_score(y_test, y_test_pred, average="weighted"))
    test_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))
    test_roc_auc = float(roc_auc_score(y_test, y_test_proba, multi_class="ovr", average="weighted"))

    print(f"      -> Acurácia (Validação): {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"      -> Acurácia (Teste)    : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"      -> Precisão (Weighted) : {test_precision:.4f}")
    print(f"      -> Recall (Weighted)   : {test_recall:.4f}")
    print(f"      -> F1-Score (Weighted) : {test_f1:.4f}")
    print(f"      -> ROC-AUC (OVR)       : {test_roc_auc:.4f}")
    print("\nRelatório de Classificação por Classe (Conjunto de Teste):")
    print(classification_report(y_test, y_test_pred, digits=4))

    # 5. Serialização do Artefato e Relatório de Métricas
    model_artifact_path = MODELS_DIR / "xgboost_mnist.joblib"
    metrics_json_path = MODELS_DIR / "metrics_report.json"

    print(f"[5/5] Serializando modelo em '{model_artifact_path.relative_to(PROJECT_ROOT)}'...")
    joblib.dump(model, model_artifact_path, compress=3)

    metrics_payload = {
        "modelo": "XGBClassifier (MNIST 784)",
        "origem_modelo": "Opção A - Reaproveitamento e Refinamento do Modelo Campeão do Mini-Projeto 2 (SCTEC)",
        "dataset": "mnist_784 (OpenML v1 - 70.000 amostras 28x28)",
        "divisao_dados": {
            "treino": int(X_train.shape[0]),
            "validacao": int(X_val.shape[0]),
            "teste": int(X_test.shape[0])
        },
        "hiperparametros": {
            "n_estimators": 200,
            "max_depth": 7,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": 42
        },
        "metricas_validacao": {
            "acuracia": round(val_acc, 4)
        },
        "metricas_teste": {
            "acuracia": round(test_acc, 4),
            "precisao_weighted": round(test_precision, 4),
            "recall_weighted": round(test_recall, 4),
            "f1_score_weighted": round(test_f1, 4),
            "roc_auc_ovr_weighted": round(test_roc_auc, 4)
        },
        "tempo_treinamento_segundos": round(train_time_sec, 2),
        "tempo_predicao_teste_segundos": round(pred_time_sec, 4),
        "matriz_confusao_teste": confusion_matrix(y_test, y_test_pred).tolist()
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4, ensure_ascii=False)

    print(f"      Relatório de métricas salvo em '{metrics_json_path.relative_to(PROJECT_ROOT)}'.")
    print("=" * 80)
    print("PIPELINE OFFLINE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    train_and_serialize_pipeline()
