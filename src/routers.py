"""
# Fase 2 e Fase 5: Módulos de Rota e Tratamento Semântico de Exceções HTTP (src/routers.py)
Agrupa os endpoints de monitoramento (`GET /` e `GET /health`) e de inferência sob demanda
(`POST /predict` via UploadFile e `POST /v1/predict` via JSON validado por Pydantic),
mapeando códigos HTTP semânticos (200 OK, 400 Bad Request, 422 Unprocessable Entity, 500/503).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from src.schemas import HealthResponse, PredictResponse, PixelPredictRequest, ErrorResponse
from src.services import inference_service

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/webp",
    "application/octet-stream",
}
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@router.get(
    "/",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Monitoramento"],
    summary="Verificação de Status da API (Rota Raiz)",
    description="Rota principal de verificação de status da API e confirmação do carregamento do modelo em memória RAM."
)
async def root_status() -> HealthResponse:
    health_data = inference_service.check_health()
    return HealthResponse(**health_data)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Monitoramento"],
    summary="Monitoramento de Saúde Operacional (/health)",
    description=(
        "Rota padronizada de checagem de integridade operacional (Fase 5). "
        "Responde com HTTP 200 OK informando se o serviço está ativo e se o modelo reside na memória RAM."
    )
)
async def health_check() -> HealthResponse:
    health_data = inference_service.check_health()
    return HealthResponse(**health_data)


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inferência Preditiva"],
    summary="Inferência de Dígito Manuscrito via Upload de Imagem (PNG/JPG)",
    responses={
        status.HTTP_200_OK: {"model": PredictResponse, "description": "Predição realizada com sucesso."},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "Arquivo enviado inválido ou corrompido."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Erro de validação dos parâmetros de entrada."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Erro interno na inferência."},
    },
    description=(
        "Recebe um arquivo de imagem (`UploadFile` PNG/JPG), executa o pipeline de pré-processamento de `src/predict.py` "
        "(conversão para escala de cinza, inversão automática de fundo claro para o padrão MNIST, redimensionamento "
        "para 28x28, achatamento para 784 posições e normalização por 255.0) e retorna o dígito previsto (0 a 9) "
        "junto com sua probabilidade de confiança."
    )
)
async def predict_image(
    file: UploadFile = File(..., description="Arquivo de imagem (PNG ou JPG) contendo um dígito manuscrito de 0 a 9.")
) -> PredictResponse:
    # Validação semântica de extensão e tipo MIME (400 Bad Request para arquivos que não sejam imagem)
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    if filename and not filename.endswith(ALLOWED_EXTENSIONS) and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Formato de arquivo não suportado ('{file.filename}'). "
                "Por favor, envie uma imagem nos formatos PNG ou JPG."
            )
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="O arquivo de imagem enviado está vazio (0 bytes)."
            )
        return inference_service.predict_from_image_bytes(image_bytes)

    except HTTPException:
        raise
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        ) from val_err
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(run_err)
        ) from run_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro inesperado ao processar a inferência da imagem: {str(exc)}"
        ) from exc


@router.post(
    "/v1/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inferência Preditiva"],
    summary="Inferência de Dígito Manuscrito via Payload JSON Estrito (784 pixels)",
    responses={
        status.HTTP_200_OK: {"model": PredictResponse, "description": "Predição realizada com sucesso."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Dados inválidos interceptados pelo Pydantic (ex: tamanho != 784 ou valores fora de [0.0, 1.0])."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Erro interno na inferência."},
    },
    description=(
        "Consome um esquema JSON validado estritamente via Pydantic (`PixelPredictRequest`), exigindo exatamente "
        "784 números de ponto flutuante entre 0.0 e 1.0. Caso tipos incorretos ou valores fora da faixa sejam enviados, "
        "o Pydantic intercepta automaticamente com HTTP 422 Unprocessable Entity antes de acionar o modelo."
    )
)
async def predict_json_pixels(payload: PixelPredictRequest) -> PredictResponse:
    try:
        return inference_service.predict_from_pixel_schema(payload)
    except RuntimeError as run_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(run_err)
        ) from run_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno durante a inferência via vetor JSON: {str(exc)}"
        ) from exc
