# Imagens de Amostra (`data/samples/`)

Esta pasta contém imagens de dígitos manuscritos (`PNG` e `JPG`) para teste imediato nos endpoints da API FastAPI (`POST /predict`) e na interface cliente Streamlit (`app.py`):

- Imagens com fundo preto e traço branco (padrão nativo MNIST `28x28`).
- Imagens com fundo claro (papel branco) e escrita escura, demonstrando a detecção e inversão automática de cores (`255 - pixel`) implementada em `src/predict.py`.
- Imagens manuscritas digitalizadas do Desafio C (`um.jpg`, `dois.jpg`, `três.jpg`, etc.).
