"""
# Fase 2 e Fase 3: Ponto de Entrada da API FastAPI e Ciclo de Vida Otimizado (src/main.py)
Configura a instância principal do FastAPI, gerencia o ciclo de vida (`@asynccontextmanager` lifespan)
garantindo que o modelo XGBoost (`models/xgboost_mnist.joblib`) seja carregado na memória RAM uma única
vez na inicialização do servidor ASGI (Uvicorn), e registra os roteadores modulares.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.routers import router as api_router
from src.services import inference_service

# Configuração de logs estruturados
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("mnist_xgboost_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    # Fase 3: Ciclo de Vida e Inicialização Otimizada (Lifespan)
    Carrega o artefato serializado do modelo XGBoost na memória RAM exatamente uma vez
    durante o startup da aplicação FastAPI e libera os recursos no shutdown.
    """
    logger.info("Iniciando ciclo de vida (lifespan) da API MNIST XGBoost...")
    try:
        inference_service.load_model_into_ram()
        app.state.model_loaded = True
    except Exception as exc:
        logger.error("Falha crítica ao carregar modelo no startup: %s", exc)
        app.state.model_loaded = False
        raise

    yield

    logger.info("Encerrando aplicação e limpando artefato da memória RAM...")
    inference_service.unload_model_from_ram()
    app.state.model_loaded = False


app = FastAPI(
    title="MNIST Handwritten Digit Classifier API - XGBoost (SCTEC)",
    description=(
        "Microsserviço preditivo RESTful construído com **FastAPI**, **Pydantic** e **XGBoost** "
        "para classificação de dígitos manuscritos (0 a 9) baseados no dataset **MNIST (`mnist_784`)**.\n\n"
        "### Arquitetura em Camadas e Requisitos MLOps Atendidos:\n"
        "- **Fase 1**: Pipeline Offline treinado com divisão estratificada de Treino/Validação/Teste e serializado em `models/xgboost_mnist.joblib`.\n"
        "- **Fase 2**: Arquitetura Modular dividida em `src/schemas.py`, `src/services.py`, `src/routers.py`, `src/predict.py` e `src/main.py`.\n"
        "- **Fase 3**: Carregamento único em memória RAM via `@asynccontextmanager` (`lifespan`).\n"
        "- **Fase 4**: Validação estrita de contratos de entrada e saída via Pydantic (`BaseModel`, `Field`).\n"
        "- **Fase 5**: Monitoramento em `GET /` e `GET /health` e inferência em `POST /predict` (Upload de Imagem) e `POST /v1/predict` (Payload JSON)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Habilita CORS para integração limpa com o Front-End Streamlit ou clientes externos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handler global de segurança para evitar crashes não tratados (HTTP 500 estruturado).
    """
    logger.exception("Exceção não tratada na rota %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "erro": "Erro Interno do Servidor",
            "detalhe": f"Ocorreu uma falha inesperada ao processar a requisição: {str(exc)}",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
        }
    )


# Inclusão dos roteadores modulares de negócio
app.include_router(api_router)
