"""
# Requisito 1: Pipeline de Pré-processamento de Imagens (src/predict.py)
Módulo responsável por converter a imagem carregada (via Upload OpenCV/PIL) para escala de cinza,
redimensionar para 28x28 pixels, inverter cores automaticamente caso o fundo seja claro e a escrita escura
(garantindo compatibilidade estrita com o padrão MNIST de fundo preto e traço branco), achatar para um
vetor unidimensional de 784 posições e normalizar dividindo por 255.0.
"""

import io
from typing import Tuple, Union
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError


class MNISTImagePreprocessor:
    """
    Classe utilitária de pré-processamento de imagens para o modelo MNIST (28x28 = 784 características).
    Suporta carregamento via bytes brutos (UploadFile), objetos PIL.Image ou matrizes NumPy/OpenCV.
    """

    TARGET_SIZE: Tuple[int, int] = (28, 28)
    EXPECTED_FEATURES: int = 784

    @staticmethod
    def preprocess_from_bytes(
        image_bytes: bytes,
        auto_invert: bool = True,
        center_digit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Converte bytes de uma imagem enviada via HTTP Upload em um vetor (1, 784) normalizado [0.0, 1.0].

        Etapas do pipeline:
        1. Decodificação segura da imagem via PIL / OpenCV e conversão para escala de cinza ('L').
        2. Detecção inteligente de fundo claro (média das bordas/imagem > 127) e inversão (255 - pixel)
           para assegurar fundo preto (0) e escrita branca (255) no padrão MNIST.
        3. Remoção opcional de ruído de fundo e centralização por caixa delimitadora (bounding box)
           quando a imagem possui alta resolução (ex: foto ou desenho manuscrito em folha branca),
           preservando imagens que já estão em 28x28.
        4. Redimensionamento para 28x28 pixels.
        5. Achatamento (flatten) para vetor unidimensional de 784 posições (shape: 1, 784).
        6. Normalização matemática dividindo os valores por 255.0.

        Retorna:
            Tuple[np.ndarray, np.ndarray, bool]:
                - vetor_normalizado: Array NumPy float32 com shape (1, 784) no intervalo [0.0, 1.0].
                - img_28x28: Matriz NumPy uint8 (28, 28) após redimensionamento/inversão (para visualização).
                - inversao_aplicada: Booleano indicando se a inversão de cores foi acionada.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("O arquivo enviado está vazio (0 bytes).")

        # 1. Carregamento via PIL e conversão para Escala de Cinza (1 canal, 0-255)
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
            gray_img = np.array(pil_img, dtype=np.uint8)
        except (UnidentifiedImageError, OSError) as exc:
            # Fallback via OpenCV caso o cabeçalho de bytes exija imdecode direto
            np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(np_buffer, cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise ValueError("Não foi possível decodificar o arquivo como uma imagem PNG/JPG válida.") from exc
            gray_img = decoded

        if gray_img.size == 0:
            raise ValueError("A imagem decodificada possui dimensões inválidas.")

        orig_h, orig_w = gray_img.shape[:2]

        # 2. Inversão automática de cores se o fundo for claro (padrão folha branca / traço escuro)
        inversao_aplicada = False
        if auto_invert:
            # Calcula a intensidade média das bordas externas e a média global da imagem
            border_pixels = np.concatenate([
                gray_img[0, :],
                gray_img[-1, :],
                gray_img[:, 0],
                gray_img[:, -1]
            ])
            border_mean = float(np.mean(border_pixels))
            global_mean = float(np.mean(gray_img))

            # Se a borda for predominantemente clara (> 127.0) ou a média geral for clara (> 127.0),
            # invertemos os canais para ficar compatível com o MNIST (fundo preto = 0, dígito claro = 255)
            if border_mean > 127.0 or global_mean > 127.0:
                gray_img = 255 - gray_img
                inversao_aplicada = True

        # 3. Tratamento e Redimensionamento para 28x28 pixels
        # Se a imagem já for 28x28 (ex: amostras exportadas do próprio MNIST ou redimensionadas previamente),
        # mantemos a proporção exata. Caso seja uma foto/desenho maior com fundo ruidoso, limpamos o contraste
        # e redimensionamos preservando a nitidez do traço.
        if (orig_h, orig_w) == MNISTImagePreprocessor.TARGET_SIZE:
            img_28x28 = gray_img
        else:
            if center_digit and (orig_h > 32 or orig_w > 32):
                img_28x28 = MNISTImagePreprocessor._clean_and_resize_high_res(gray_img)
            else:
                img_28x28 = cv2.resize(
                    gray_img,
                    MNISTImagePreprocessor.TARGET_SIZE,
                    interpolation=cv2.INTER_AREA
                )

        # 4. Achatar para vetor unidimensional de 784 posições (shape 1x784) e normalizar por 255.0
        vetor_784 = img_28x28.reshape(1, MNISTImagePreprocessor.EXPECTED_FEATURES).astype(np.float32)
        vetor_normalizado = vetor_784 / 255.0

        # Garante limites estritos em [0.0, 1.0]
        vetor_normalizado = np.clip(vetor_normalizado, 0.0, 1.0)

        return vetor_normalizado, img_28x28, inversao_aplicada

    @staticmethod
    def _clean_and_resize_high_res(gray_inverted: np.ndarray) -> np.ndarray:
        """
        Auxiliar para fotos de dígitos manuscritos (como um_80.jpg, dois.jpg do Desafio C):
        Remove sombras leves do papel (limiarização suave), centraliza o dígito em um quadro
        proporcional com margem (padrão 20x20 centralizado em 28x28 do MNIST) quando há contorno
        nítido, ou redimensiona com INTER_AREA.
        """
        # Subtrai o nível de fundo (percentil 50 das bordas) para eliminar cinza de iluminação em fotos de papel
        border = np.concatenate([
            gray_inverted[0, :], gray_inverted[-1, :],
            gray_inverted[:, 0], gray_inverted[:, -1]
        ])
        bg_level = np.percentile(border, 75)
        cleaned = np.clip(gray_inverted.astype(np.float32) - bg_level, 0, 255).astype(np.uint8)

        # Normaliza o contraste se houver sinal relevante
        max_val = int(np.max(cleaned))
        if max_val > 25:
            cleaned = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)
            # Encontra bounding box dos pixels ativos (> 30)
            coords = cv2.findNonZero((cleaned > 30).astype(np.uint8))
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                # Só recorta se a caixa delimitadora não for apenas um ruído puntiforme
                if w >= 4 and h >= 4:
                    cropped = cleaned[y:y + h, x:x + w]
                    # Redimensiona o maior lado para 20 pixels mantendo aspect ratio (padrão oficial MNIST)
                    scale = 20.0 / max(w, h)
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    resized_digit = cv2.resize(
                        cropped,
                        (new_w, new_h),
                        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                    )
                    # Insere centralizado em um canvas 28x28 preto
                    canvas = np.zeros((28, 28), dtype=np.uint8)
                    pad_y = (28 - new_h) // 2
                    pad_x = (28 - new_w) // 2
                    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_digit
                    return canvas

        # Fallback: redimensionamento direto da imagem invertida original para 28x28
        return cv2.resize(
            gray_inverted,
            MNISTImagePreprocessor.TARGET_SIZE,
            interpolation=cv2.INTER_AREA
        )


def preprocess_image(
    image_input: Union[bytes, Image.Image, np.ndarray],
    auto_invert: bool = True
) -> np.ndarray:
    """
    Função direta de pré-processamento conforme especificado no Requisito 1:
    Converte a imagem carregada para escala de cinza, redimensiona para 28x28,
    inverte cores se o fundo for claro, achata para vetor unidimensional de 784 posições
    e normaliza dividindo por 255.0.

    Retorna:
        np.ndarray: Vetor 2D de shape (1, 784) pronto para `model.predict(X)` e `model.predict_proba(X)`.
    """
    if isinstance(image_input, bytes):
        vetor_norm, _, _ = MNISTImagePreprocessor.preprocess_from_bytes(
            image_input, auto_invert=auto_invert
        )
        return vetor_norm

    if isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="PNG")
        vetor_norm, _, _ = MNISTImagePreprocessor.preprocess_from_bytes(
            buf.getvalue(), auto_invert=auto_invert
        )
        return vetor_norm

    if isinstance(image_input, np.ndarray):
        success, encoded = cv2.imencode(".png", image_input)
        if not success:
            raise ValueError("Falha ao codificar matriz NumPy de entrada.")
        vetor_norm, _, _ = MNISTImagePreprocessor.preprocess_from_bytes(
            encoded.tobytes(), auto_invert=auto_invert
        )
        return vetor_norm

    raise TypeError(f"Tipo de entrada não suportado para pré-processamento: {type(image_input)}")
