# Diretório de Modelos Serializados (`models/`)

Esta pasta armazena os artefatos binários gerados na **Fase 1 (Pipeline Offline)**:

- `xgboost_mnist.joblib`: Modelo classificador **XGBoost (`XGBClassifier`)** treinado com 784 features do dataset `mnist_784` (28x28 pixels normalizados em `[0.0, 1.0]`).
- `metrics_report.json`: Relatório consolidado de métricas alcançadas nos conjuntos de validação e teste (Acurácia, Precisão, Recall, F1-Score, ROC-AUC e Matriz de Confusão).

Para regerar ou atualizar os artefatos a qualquer momento, execute na raiz do projeto:

```bash
python -m src.train
```
