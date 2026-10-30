"""
# Fase Front-End (app.py) - Cliente Web Streamlit
Interface de usuário interativa que atua estritamente como cliente HTTP consumindo o microsserviço
preditivo FastAPI (`http://127.0.0.1:8000/predict` e `http://127.0.0.1:8000/health`) utilizando
a biblioteca `requests`.

Permite:
1. Verificar em tempo real o status da API FastAPI e se o modelo XGBoost reside em memória RAM.
2. Fazer upload de arquivos de imagem (PNG/JPG) ou selecionar imagens reais de teste da pasta `data/samples/`
   (incluindo os dígitos manuscritos próprios digitalizados).
3. Enviar a imagem via requisição HTTP POST (`requests.post`) para `http://127.0.0.1:8000/predict`.
4. Exibir a imagem enviada, o JSON retornado pela FastAPI, o dígito previsto (0 a 9), o percentual de
   confiança e o gráfico de distribuição de probabilidades entre as 10 classes.
"""

import io
import json
from pathlib import Path
import requests
import streamlit as st
from PIL import Image
import pandas as pd

# Configurações de endereço da API Back-End FastAPI
API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics_report.json"

st.set_page_config(
    page_title="Classificador MNIST - XGBoost & FastAPI (SCTEC)",
    page_icon="🔢",
    layout="wide"
)

# Cabeçalho Principal
st.title("🔢 Classificador de Dígitos Manuscritos (MNIST) com XGBoost")
st.markdown(
    "**Projeto Avaliativo Final — SCTEC | Arquitetura de Microsserviços (Streamlit Front-End + FastAPI Back-End + XGBoost)**"
)
st.divider()

# Barra Lateral: Monitoramento de Saúde (/health) e Métricas do Modelo
with st.sidebar:
    st.header("📡 Status do Microsserviço")
    st.caption(f"Endpoint: `{HEALTH_ENDPOINT}`")

    if st.button("🔄 Verificar Saúde da API (/health)", use_container_width=True):
        st.rerun()

    api_online = False
    try:
        health_resp = requests.get(HEALTH_ENDPOINT, timeout=3)
        if health_resp.status_code == 200:
            health_json = health_resp.json()
            api_online = health_json.get("modelo_carregado_ram", False)
            if api_online:
                st.success("✅ API Online & Modelo em RAM")
            else:
                st.warning("⚠️ API Online, mas Modelo ausente da RAM")
            with st.expander("Ver JSON de `/health`"):
                st.json(health_json)
        else:
            st.error(f"❌ Status HTTP inesperado: {health_resp.status_code}")
    except requests.exceptions.RequestException:
        st.error("❌ API FastAPI Offline")
        st.info(
            "Inicie o servidor Back-End no terminal:\n"
            "```bash\n"
            "uvicorn src.main:app --reload\n"
            "```"
        )

    st.divider()
    st.subheader("📊 Métricas Offline do Modelo")
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            test_m = metrics_data.get("metricas_teste", {})
            c1, c2 = st.columns(2)
            c1.metric("Acurácia (Teste)", f"{test_m.get('acuracia', 0)*100:.2f}%")
            c2.metric("F1-Score", f"{test_m.get('f1_score_weighted', 0)*100:.2f}%")
            c3, c4 = st.columns(2)
            c3.metric("Precisão", f"{test_m.get('precisao_weighted', 0)*100:.2f}%")
            c4.metric("ROC-AUC", f"{test_m.get('roc_auc_ovr_weighted', 0)*100:.2f}%")
        except Exception:
            st.caption("Não foi possível ler o arquivo de métricas.")
    else:
        st.caption("Arquivo `models/metrics_report.json` ainda não gerado.")

    st.divider()
    st.markdown("🔗 **Documentação Interativa:**")
    st.markdown(f"- [Swagger UI (`/docs`)]({API_BASE_URL}/docs)")
    st.markdown(f"- [ReDoc (`/redoc`)]({API_BASE_URL}/redoc)")

# Layout Principal em duas colunas: Entrada da Imagem (Esquerda) e Resultado da Inferência (Direita)
col_input, col_result = st.columns([1, 1.2], gap="large")

image_bytes_to_send = None
selected_filename = None

with col_input:
    st.subheader("📤 1. Envio da Imagem do Dígito")

    modo_entrada = st.radio(
        "Escolha a fonte da imagem:",
        options=[
            "Fazer Upload de Arquivo (PNG/JPG)",
            "Usar Imagens de Exemplo do Projeto (data/samples/)"
        ],
        horizontal=False
    )

    if modo_entrada == "Fazer Upload de Arquivo (PNG/JPG)":
        uploaded_file = st.file_uploader(
            "Selecione uma imagem de dígito manuscrito (0 a 9):",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            image_bytes_to_send = uploaded_file.getvalue()
            selected_filename = uploaded_file.name
    else:
        if SAMPLES_DIR.exists():
            sample_files = sorted(
                [p.name for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            )
            if sample_files:
                escolhido = st.selectbox("Selecione uma imagem de amostra:", sample_files)
                sample_path = SAMPLES_DIR / escolhido
                image_bytes_to_send = sample_path.read_bytes()
                selected_filename = escolhido
            else:
                st.warning("Nenhuma imagem encontrada em `data/samples/`.")
        else:
            st.warning("Pasta `data/samples/` não encontrada.")

    if image_bytes_to_send is not None:
        st.markdown("#### 🖼️ Pré-visualização da Imagem Enviada")
        img_preview = Image.open(io.BytesIO(image_bytes_to_send))
        p_col1, p_col2 = st.columns([1, 1])
        with p_col1:
            st.image(img_preview, caption=f"Arquivo: {selected_filename}", width=180)
        with p_col2:
            st.caption(f"**Dimensões originais:** {img_preview.size[0]} x {img_preview.size[1]} px")
            st.caption(f"**Modo de cor:** {img_preview.mode}")
            st.caption(
                "O pipeline `src/predict.py` converterá para escala de cinza, "
                "inverterá automaticamente se o fundo for claro, redimensionará para 28x28 "
                "e normalizará o vetor de 784 posições por 255.0."
            )

        btn_predict = st.button(
            "🚀 Classificar Dígito via API FastAPI",
            type="primary",
            use_container_width=True
        )
    else:
        btn_predict = False
        st.info("Envie uma imagem PNG/JPG acima para habilitar a classificação.")

with col_result:
    st.subheader("🎯 2. Resultado da Predição (FastAPI + XGBoost)")

    if image_bytes_to_send is not None and btn_predict:
        with st.spinner("Enviando requisição HTTP POST para http://127.0.0.1:8000/predict ..."):
            try:
                files_payload = {
                    "file": (selected_filename or "digito.png", image_bytes_to_send, "image/png")
                }
                response = requests.post(PREDICT_ENDPOINT, files=files_payload, timeout=10)

                if response.status_code == 200:
                    res_json = response.json()
                    digito_previsto = res_json["predicao"]
                    probabilidade = res_json["probabilidade"]
                    confianca_pct = res_json.get("confianca_percentual", round(probabilidade * 100, 2))
                    inversao = res_json.get("inversao_aplicada", False)
                    tempo_ms = res_json.get("tempo_inferencia_ms", 0.0)

                    # Cards de Destaque do Resultado
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🔢 Dígito Previsto", f"{digito_previsto}")
                    m2.metric("📈 Confiança", f"{confianca_pct:.2f}%")
                    m3.metric("⚡ Latência Inferência", f"{tempo_ms:.2f} ms")

                    if inversao:
                        st.info(
                            "🔄 **Inversão Automática Aplicada:** Foi detectado fundo claro e escrita escura; "
                            "as cores foram invertidas automaticamente para o padrão MNIST (fundo preto e traço branco)."
                        )

                    # Gráfico de Barras com as probabilidades das 10 classes
                    probs_dict = res_json.get("probabilidades_classes", {})
                    if probs_dict:
                        st.markdown("#### 📊 Distribuição de Probabilidades por Classe (0 a 9)")
                        df_probs = pd.DataFrame({
                            "Dígito": list(probs_dict.keys()),
                            "Probabilidade (%)": [round(v * 100.0, 2) for v in probs_dict.values()]
                        }).set_index("Dígito")
                        st.bar_chart(df_probs)

                    # Exibição do JSON bruto retornado pela API FastAPI
                    st.markdown("#### 📦 Payload JSON Retornado pela FastAPI (`POST /predict`)")
                    st.json(res_json)

                else:
                    st.error(f"Erro HTTP {response.status_code} retornado pela API:")
                    try:
                        st.json(response.json())
                    except Exception:
                        st.text(response.text)

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Não foi possível conectar ao servidor FastAPI em `http://127.0.0.1:8000/predict`. "
                    "Certifique-se de que o Back-End está em execução (`uvicorn src.main:app --reload`)."
                )
            except Exception as exc:
                st.error(f"Falha ao processar requisição: {exc}")
    else:
        st.caption("Aguardando envio de imagem e clique no botão **Classificar Dígito via API FastAPI**.")
