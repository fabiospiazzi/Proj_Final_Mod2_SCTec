"""
# Fase 4: Modelagem e Validação Estrita de Dados (Pydantic)
Módulo `src/schemas.py` responsável por definir explicitamente os contratos de entrada e saída
da API RESTful utilizando Pydantic (`BaseModel`, `Field`, `field_validator`), garantindo
validação estrita em tempo de execução e retorno automático de HTTP 422 Unprocessable Entity
para payloads inválidos antes de acionar o modelo preditivo.
"""

from typing import Dict, List, Annotated
from pydantic import BaseModel, Field, field_validator, ConfigDict


class HealthResponse(BaseModel):
    """
    Esquema de saída padronizado para o endpoint de monitoramento de saúde (GET /health e GET /).
    Informa o estado operacional do serviço e se o artefato de Machine Learning reside em memória RAM.
    """
    status: str = Field(
        default="ok",
        min_length=2,
        max_length=50,
        description="Status operacional do microsserviço FastAPI (ex: 'ok', 'degraded').",
        examples=["ok"]
    )
    servico_ativo: bool = Field(
        default=True,
        description="Indica se a API web está ativa e pronta para receber requisições HTTP."
    )
    modelo_carregado_ram: bool = Field(
        ...,
        description="Indica se o modelo preditivo XGBoost foi carregado na memória RAM durante o ciclo de vida (lifespan)."
    )
    caminho_artefato: str = Field(
        ...,
        min_length=3,
        max_length=255,
        description="Caminho relativo do artefato serializado do modelo em disco.",
        examples=["models/xgboost_mnist.joblib"]
    )
    dimensao_esperada: int = Field(
        default=784,
        ge=784,
        le=784,
        description="Quantidade estrita de características (features) esperadas pelo modelo (28x28 = 784 pixels)."
    )
    mensagem: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Mensagem descritiva sobre a integridade operacional do servidor.",
        examples=["Serviço ativo e modelo XGBoost carregado em memória RAM com sucesso."]
    )


class PixelPredictRequest(BaseModel):
    """
    Esquema de entrada estrito para inferência via payload JSON (POST /v1/predict ou POST /predict/json).
    Valida que o vetor de entrada contenha exatamente 784 valores numéricos normalizados no intervalo [0.0, 1.0].
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "pixels": [0.0] * 784,
                "origem": "canvas_web"
            }
        }
    )

    pixels: List[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        ...,
        min_length=784,
        max_length=784,
        description=(
            "Vetor unidimensional contendo exatamente 784 valores de intensidade de pixel (imagem 28x28 achatada), "
            "normalizados estritamente entre 0.0 (fundo preto MNIST) e 1.0 (traço branco MNIST)."
        )
    )
    origem: str = Field(
        default="api_json_client",
        min_length=2,
        max_length=100,
        description="Identificador textual da aplicação cliente que originou a requisição.",
        examples=["streamlit_frontend", "swagger_ui_test"]
    )

    @field_validator("pixels")
    @classmethod
    def validar_dimensao_e_valores(cls, v: List[float]) -> List[float]:
        if len(v) != 784:
            raise ValueError(f"O vetor de características deve conter exatamente 784 posições (recebido: {len(v)}).")
        return v


class PredictResponse(BaseModel):
    """
    Esquema de saída padronizado para os endpoints de inferência (POST /predict e POST /v1/predict).
    Contém obrigatoriamente `predicao` e `probabilidade`, além de metadados detalhados de confiança por classe.
    """
    predicao: int = Field(
        ...,
        ge=0,
        le=9,
        description="O número do dígito manuscrito previsto pelo classificador XGBoost (0 a 9).",
        examples=[2]
    )
    probabilidade: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A probabilidade/confiança associada à classe prevista (intervalo de 0.0 a 1.0).",
        examples=[0.9875]
    )
    confianca_percentual: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentual de confiança formatado de 0.0 a 100.0%.",
        examples=[98.75]
    )
    probabilidades_classes: Dict[str, float] = Field(
        ...,
        description="Dicionário contendo a distribuição de probabilidade para cada um dos 10 dígitos ('0' a '9')."
    )
    inversao_aplicada: bool = Field(
        default=False,
        description="Indica se a imagem original tinha fundo claro e teve suas cores invertidas automaticamente para o padrão MNIST."
    )
    tempo_inferencia_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo gasto exclusivamente no pré-processamento e inferência matemática em milissegundos.",
        examples=[4.32]
    )


class ErrorResponse(BaseModel):
    """
    Esquema padronizado para retorno de exceções HTTP tratadas (400, 422, 500, 503).
    """
    erro: str = Field(
        ...,
        min_length=3,
        max_length=120,
        description="Título semântico do erro HTTP ocorrido.",
        examples=["Arquivo de Imagem Inválido"]
    )
    detalhe: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Explicação clara do motivo do erro e como corrigi-lo.",
        examples=["O arquivo enviado não pôde ser decodificado como uma imagem PNG ou JPG válida."]
    )
    status_code: int = Field(
        ...,
        ge=400,
        le=599,
        description="Código de status HTTP retornado.",
        examples=[400]
    )
