"""
# Fase 2 e Fase 3: Camada de Serviço e Gerenciamento do Modelo em Memória RAM (src/services.py)
Módulo responsável por carregar o artefato serializado (`models/xgboost_mnist.joblib`) uma única vez
na inicialização da aplicação (via `@asynccontextmanager` lifespan em `src/main.py`) e executar a
inferência matemática desacoplada do protocolo HTTP.

PROIBIÇÃO TÉCNICA CUMPRIDA: O modelo nunca é recarregado do disco dentro das rotas a cada requisição.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np

from src.predict import MNISTImagePreprocessor
from src.schemas import PredictResponse, PixelPredictRequest

logger = logging.getLogger("mnist_xgboost_api")

# Caminho relativo padrão para o artefato serializado do modelo (Fase 5.1: proibido caminhos absolutos fixos)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_REL_PATH = "models/xgboost_mnist.joblib"
DEFAULT_MODEL_PATH = PROJECT_ROOT / DEFAULT_MODEL_REL_PATH


class ModelInferenceService:
    """
    Serviço Singleton de Inferência do Modelo XGBoost para Classificação MNIST.
    Mantém a referência do classificador exclusivamente em memória RAM durante todo o ciclo de vida da API.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path: Path = model_path
        self.model: Optional[Any] = None
        self.is_loaded: bool = False

    def load_model_into_ram(self) -> None:
        """
        Carrega o arquivo binário `.joblib` para a memória RAM uma única vez (invocado no startup do Lifespan).
        """
        if not self.model_path.exists():
            logger.error("Artefato do modelo não encontrado em: %s", self.model_path)
            self.model = None
            self.is_loaded = False
            raise FileNotFoundError(
                f"Arquivo do modelo serializado não encontrado em '{self.model_path.relative_to(PROJECT_ROOT)}'. "
                "Execute `python -m src.train` antes de iniciar a API."
            )

        start_t = time.perf_counter()
        self.model = joblib.load(self.model_path)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.is_loaded = True
        logger.info(
            "Modelo XGBoost carregado na memória RAM com sucesso em %.2f ms (%s).",
            elapsed_ms,
            DEFAULT_MODEL_REL_PATH
        )

    def unload_model_from_ram(self) -> None:
        """
        Libera os recursos da memória RAM no encerramento da aplicação (shutdown do Lifespan).
        """
        self.model = None
        self.is_loaded = False
        logger.info("Modelo XGBoost removido da memória RAM no encerramento do serviço.")

    def check_health(self) -> Dict[str, Any]:
        """
        Retorna os dados de integridade operacional do modelo em memória RAM para o endpoint `/health`.
        """
        loaded = self.is_loaded and (self.model is not None)
        return {
            "status": "ok" if loaded else "degraded",
            "servico_ativo": True,
            "modelo_carregado_ram": loaded,
            "caminho_artefato": DEFAULT_MODEL_REL_PATH,
            "dimensao_esperada": 784,
            "mensagem": (
                "Serviço ativo e modelo XGBoost residente corretamente na memória RAM."
                if loaded
                else "Serviço ativo, porém o modelo preditivo não está carregado na memória RAM."
            )
        }

    def predict_from_image_bytes(self, image_bytes: bytes) -> PredictResponse:
        """
        Executa o pipeline completo de pré-processamento de imagem e inferência matemática usando o modelo em RAM.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("O modelo XGBoost não está carregado na memória RAM.")

        start_t = time.perf_counter()

        # 1. Pipeline de Pré-processamento de `src/predict.py`
        vetor_normalizado, _, inversao_aplicada = MNISTImagePreprocessor.preprocess_from_bytes(
            image_bytes=image_bytes,
            auto_invert=True,
            center_digit=True
        )

        # 2. Inferência Matemática com o modelo residente em RAM
        probabilities = self.model.predict_proba(vetor_normalizado)[0]
        predicted_class_idx = int(np.argmax(probabilities))

        # Mapeia para o rótulo real da classe caso classes_ esteja definido no estimador
        if hasattr(self.model, "classes_"):
            predicted_digit = int(self.model.classes_[predicted_class_idx])
            prob_dict = {
                str(int(cls_label)): round(float(prob), 6)
                for cls_label, prob in zip(self.model.classes_, probabilities)
            }
        else:
            predicted_digit = predicted_class_idx
            prob_dict = {str(i): round(float(prob), 6) for i, prob in enumerate(probabilities)}

        max_prob = round(float(probabilities[predicted_class_idx]), 6)
        elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 3)

        return PredictResponse(
            predicao=predicted_digit,
            probabilidade=max_prob,
            confianca_percentual=round(max_prob * 100.0, 2),
            probabilidades_classes=prob_dict,
            inversao_aplicada=inversao_aplicada,
            tempo_inferencia_ms=elapsed_ms
        )

    def predict_from_pixel_schema(self, payload: PixelPredictRequest) -> PredictResponse:
        """
        Executa a inferência matemática a partir de um vetor JSON de 784 pixels já validado pelo Pydantic.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("O modelo XGBoost não está carregado na memória RAM.")

        start_t = time.perf_counter()

        vetor_normalizado = np.array(payload.pixels, dtype=np.float32).reshape(1, 784)
        probabilities = self.model.predict_proba(vetor_normalizado)[0]
        predicted_class_idx = int(np.argmax(probabilities))

        if hasattr(self.model, "classes_"):
            predicted_digit = int(self.model.classes_[predicted_class_idx])
            prob_dict = {
                str(int(cls_label)): round(float(prob), 6)
                for cls_label, prob in zip(self.model.classes_, probabilities)
            }
        else:
            predicted_digit = predicted_class_idx
            prob_dict = {str(i): round(float(prob), 6) for i, prob in enumerate(probabilities)}

        max_prob = round(float(probabilities[predicted_class_idx]), 6)
        elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 3)

        return PredictResponse(
            predicao=predicted_digit,
            probabilidade=max_prob,
            confianca_percentual=round(max_prob * 100.0, 2),
            probabilidades_classes=prob_dict,
            inversao_aplicada=False,
            tempo_inferencia_ms=elapsed_ms
        )


# Instância única compartilhada pelo ciclo de vida (lifespan) da aplicação FastAPI
inference_service = ModelInferenceService()
