"""
image_processor.py
-------------------
Pipeline de processamento de imagem para transformar uma foto comum em
um retrato profissional (estilo headshot corporativo/LinkedIn).

Princípio central de design: **a identidade e os traços faciais da
pessoa nunca são alterados**. Todas as operações abaixo atuam apenas em
aspectos periféricos:

- Remoção/substituição de fundo (rembg) — não toca nos pixels do sujeito.
- Correção de iluminação e contraste — aplicada de forma global e sutil,
  nunca via redesenho/geração de rosto.
- Redução de ruído — filtro clássico de denoising (OpenCV), não um
  modelo generativo.
- Recorte/enquadramento — apenas corta a imagem, não redesenha nada.
- Integração opcional com API generativa (Replicate/Stability) — usada,
  quando configurada, SOMENTE para gerar/inpaintar o fundo a partir de
  uma máscara que protege 100% da área do sujeito, nunca do rosto.

Todas as funções são independentes e tipadas, para facilitar testes e
reuso fora do Streamlit.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from config import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceções específicas do domínio
# --------------------------------------------------------------------------- #

class ImageProcessingError(Exception):
    """Erro genérico durante o processamento da imagem."""


class InvalidImageError(ImageProcessingError):
    """A imagem enviada é inválida (formato, tamanho ou dimensões)."""


class NoFaceDetectedError(ImageProcessingError):
    """Nenhum rosto foi identificado na imagem enviada."""


class BackgroundRemovalError(ImageProcessingError):
    """Falha ao remover o fundo da imagem (rembg/onnxruntime)."""


class GenerativeAPIError(ImageProcessingError):
    """Falha na chamada opcional à API generativa externa."""


# --------------------------------------------------------------------------- #
# Estrutura de resultado do rosto detectado
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FaceBox:
    """Retângulo delimitador de um rosto detectado, em pixels."""

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


# --------------------------------------------------------------------------- #
# 1. Validação de entrada
# --------------------------------------------------------------------------- #

def validate_image(file_bytes: bytes, filename: str) -> Image.Image:
    """Valida os bytes de upload e retorna uma imagem PIL em modo RGB.

    Args:
        file_bytes: Conteúdo bruto do arquivo enviado pelo usuário.
        filename: Nome original do arquivo (usado para checar a extensão).

    Returns:
        Imagem PIL já convertida para RGB.

    Raises:
        InvalidImageError: Se o formato, tamanho ou dimensões forem inválidos,
            ou se o arquivo estiver corrompido.
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in config.ALLOWED_EXTENSIONS:
        raise InvalidImageError(
            f"Formato '.{extension}' não suportado. "
            f"Envie um arquivo {', '.join(config.ALLOWED_EXTENSIONS)}."
        )

    if len(file_bytes) > config.max_upload_size_bytes:
        raise InvalidImageError(
            f"Arquivo muito grande ({len(file_bytes) / 1024 / 1024:.1f} MB). "
            f"O limite é {config.MAX_UPLOAD_SIZE_MB} MB."
        )

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # força a decodificação completa, revela corrupção
    except Exception as exc:  # noqa: BLE001 - queremos capturar qualquer falha de decodificação
        raise InvalidImageError(
            "Não foi possível abrir a imagem. O arquivo pode estar corrompido."
        ) from exc

    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    if min(width, height) < config.MIN_IMAGE_DIMENSION:
        raise InvalidImageError(
            f"Imagem muito pequena ({width}x{height}px). "
            f"Envie uma foto com pelo menos {config.MIN_IMAGE_DIMENSION}px "
            "no menor lado."
        )

    return image


# --------------------------------------------------------------------------- #
# 2. Detecção de face
# --------------------------------------------------------------------------- #

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Carrega (uma única vez) o classificador Haar Cascade de faces."""
    global _face_cascade
    if _face_cascade is None:
        cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
        if cascade.empty():
            raise ImageProcessingError(
                "Falha ao carregar o modelo de detecção facial do OpenCV."
            )
        _face_cascade = cascade
    return _face_cascade


def detect_faces(image: Image.Image) -> list[FaceBox]:
    """Detecta todos os rostos em uma imagem PIL, do maior para o menor.

    Usado tanto pelo pipeline principal quanto pela UI, que pode pedir ao
    usuário para escolher qual rosto é o dele quando há mais de uma pessoa
    na foto.

    Args:
        image: Imagem PIL em modo RGB.

    Returns:
        Lista de `FaceBox`, ordenada da maior para a menor área. Lista
        vazia se nenhum rosto for encontrado.
    """
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=config.FACE_CASCADE_SCALE_FACTOR,
        minNeighbors=config.FACE_CASCADE_MIN_NEIGHBORS,
        minSize=config.FACE_CASCADE_MIN_SIZE,
    )

    boxes = [FaceBox(x=int(x), y=int(y), w=int(w), h=int(h)) for x, y, w, h in faces]
    boxes.sort(key=lambda box: box.w * box.h, reverse=True)
    return boxes


def detect_primary_face(image: Image.Image, face_index: int = 0) -> FaceBox:
    """Detecta o(s) rosto(s) e retorna o escolhido pelo índice informado.

    Args:
        image: Imagem PIL em modo RGB.
        face_index: Índice do rosto na lista ordenada por área (0 = maior
            rosto detectado). Usado quando há múltiplas pessoas na foto e
            o usuário escolheu manualmente qual é ele na UI.

    Returns:
        A caixa delimitadora (FaceBox) do rosto escolhido.

    Raises:
        NoFaceDetectedError: Se nenhum rosto for encontrado na imagem.
    """
    faces = detect_faces(image)

    if len(faces) == 0:
        raise NoFaceDetectedError(
            "Nenhum rosto foi identificado na foto. Envie uma imagem nítida, "
            "de frente, com boa iluminação e apenas uma pessoa em destaque."
        )

    safe_index = face_index if 0 <= face_index < len(faces) else 0
    return faces[safe_index]


# --------------------------------------------------------------------------- #
# 3. Remoção de fundo (rembg) — preserva 100% o sujeito
# --------------------------------------------------------------------------- #

def create_rembg_session():
    """Cria uma sessão (modelo carregado) do `rembg` para ser reutilizada.

    Carregar o modelo U^2-Net é a parte mais lenta do pipeline. Esta
    função retorna um objeto de sessão que deve ser criado **uma única
    vez** e reaproveitado entre chamadas — na aplicação Streamlit isso é
    feito com `@st.cache_resource` (ver `app.py`), evitando recarregar o
    modelo a cada foto processada.

    Returns:
        Uma sessão `rembg` pronta para uso, ou `None` se a biblioteca não
        estiver disponível (nesse caso `remove_background` cai para o
        modo sem sessão explícita).
    """
    try:
        from rembg import new_session
    except ImportError:
        return None
    return new_session("u2net")


def remove_background(image: Image.Image, session=None) -> Image.Image:
    """Remove o fundo da imagem, retornando um PNG RGBA com alpha matting.

    Usa a biblioteca `rembg`, que segmenta o sujeito via rede neural
    (U^2-Net) sem alterar nenhum pixel dentro da máscara do sujeito —
    apenas decide o que é primeiro plano e o que é fundo.

    Args:
        image: Imagem PIL original (RGB).
        session: Sessão `rembg` já carregada (ver `create_rembg_session`).
            Se `None`, o `rembg` carrega o modelo internamente a cada
            chamada — funciona, mas é mais lento.

    Returns:
        Imagem PIL em modo RGBA, com o fundo transparente.

    Raises:
        BackgroundRemovalError: Se a segmentação falhar por qualquer motivo
            (ex.: modelo não pôde ser baixado/carregado).
    """
    try:
        from rembg import remove as rembg_remove
    except ImportError as exc:
        raise BackgroundRemovalError(
            "A biblioteca 'rembg' não está instalada corretamente."
        ) from exc

    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        if session is not None:
            result_bytes = rembg_remove(buffer.getvalue(), session=session)
        else:
            result_bytes = rembg_remove(buffer.getvalue())
        subject_rgba = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        raise BackgroundRemovalError(
            "Falha ao remover o fundo da imagem. Tente novamente com outra foto."
        ) from exc

    return subject_rgba


# --------------------------------------------------------------------------- #
# 4. Geração de fundos profissionais (sintéticos, locais, sem API)
# --------------------------------------------------------------------------- #

def _generate_studio_gray(size: tuple[int, int]) -> Image.Image:
    """Fundo de estúdio cinza neutro, com vinheta suave central."""
    width, height = size
    base = np.full((height, width, 3), 210, dtype=np.uint8)
    return _apply_radial_vignette(Image.fromarray(base), darken=25)


def _generate_studio_blue(size: tuple[int, int]) -> Image.Image:
    """Fundo de estúdio azul corporativo, com vinheta suave central."""
    width, height = size
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :] = (58, 84, 120)  # azul petróleo corporativo
    return _apply_radial_vignette(Image.fromarray(base), darken=30)


def _generate_office_blur(size: tuple[int, int]) -> Image.Image:
    """Fundo simulando um escritório desfocado (gradiente + ruído suave)."""
    width, height = size
    gradient = np.tile(
        np.linspace(200, 150, height, dtype=np.uint8).reshape(height, 1),
        (1, width),
    )
    base = np.stack([gradient, gradient, gradient + 10], axis=-1).astype(np.uint8)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 6, base.shape).astype(np.int16)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy)
    return img.filter(ImageFilter.GaussianBlur(radius=18))


def _generate_white(size: tuple[int, int]) -> Image.Image:
    """Fundo branco puro, típico de fotos de documento/currículo."""
    width, height = size
    base = np.full((height, width, 3), 250, dtype=np.uint8)
    return Image.fromarray(base)


def _generate_dark_gradient(size: tuple[int, int]) -> Image.Image:
    """Fundo em gradiente escuro premium (estilo executivo)."""
    width, height = size
    top = np.array([35, 38, 48])
    bottom = np.array([10, 11, 15])
    ramp = np.linspace(top, bottom, height).astype(np.uint8)
    base = np.repeat(ramp[:, np.newaxis, :], width, axis=1)
    return Image.fromarray(base)


def _apply_radial_vignette(image: Image.Image, darken: int) -> Image.Image:
    """Escurece sutilmente as bordas da imagem para dar efeito de estúdio."""
    width, height = image.size
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max_dist
    vignette = (dist * darken).clip(0, darken).astype(np.uint8)

    arr = np.array(image).astype(np.int16)
    for channel in range(3):
        arr[:, :, channel] = np.clip(arr[:, :, channel] - vignette, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


_BACKGROUND_GENERATORS = {
    "Estúdio cinza neutro": _generate_studio_gray,
    "Estúdio azul corporativo": _generate_studio_blue,
    "Escritório desfocado": _generate_office_blur,
    "Branco puro": _generate_white,
    "Gradiente escuro premium": _generate_dark_gradient,
}


def generate_background(style: str, size: tuple[int, int]) -> Image.Image:
    """Gera um fundo profissional sintético do estilo escolhido.

    Args:
        style: Um dos valores de `config.BACKGROUND_STYLES`.
        size: Tupla (largura, altura) do fundo a ser gerado.

    Returns:
        Imagem PIL RGB do fundo gerado.

    Raises:
        ImageProcessingError: Se o estilo solicitado não existir.
    """
    generator = _BACKGROUND_GENERATORS.get(style)
    if generator is None:
        raise ImageProcessingError(f"Estilo de fundo desconhecido: '{style}'.")
    return generator(size)


def composite_with_background(
    subject_rgba: Image.Image, background: Image.Image
) -> Image.Image:
    """Compõe o sujeito (com alpha) sobre um fundo profissional.

    A composição usa o canal alpha original do `rembg`, preservando
    integralmente os pixels do sujeito — nenhuma mistura é feita dentro
    da máscara do primeiro plano.

    Args:
        subject_rgba: Imagem RGBA do sujeito com fundo removido.
        background: Imagem RGB de fundo, do mesmo tamanho do sujeito.

    Returns:
        Imagem PIL RGB final, composta.
    """
    if background.size != subject_rgba.size:
        background = background.resize(subject_rgba.size, Image.LANCZOS)
    background_rgba = background.convert("RGBA")
    composed = Image.alpha_composite(background_rgba, subject_rgba)
    return composed.convert("RGB")


# --------------------------------------------------------------------------- #
# 5. Correção de iluminação / contraste (global, sutil, sem redesenho)
# --------------------------------------------------------------------------- #

def enhance_lighting(
    image: Image.Image, brightness: float = 1.08, contrast: float = 1.08
) -> Image.Image:
    """Aplica ajustes globais e sutis de brilho, contraste e nitidez.

    Esses ajustes operam sobre a imagem inteira de forma linear — não
    há redesenho de nenhuma região, portanto os traços faciais
    permanecem inalterados.

    Args:
        image: Imagem PIL RGB de entrada.
        brightness: Fator de brilho (1.0 = sem alteração).
        contrast: Fator de contraste (1.0 = sem alteração).

    Returns:
        Imagem PIL RGB ajustada.
    """
    result = ImageEnhance.Brightness(image).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(1.05)
    result = ImageEnhance.Sharpness(result).enhance(1.15)
    return result


def auto_white_balance(image: Image.Image) -> Image.Image:
    """Corrige o balanço de branco usando o algoritmo Gray World.

    Fotos de celular frequentemente têm um leve tom amarelado ou
    azulado por causa da luz ambiente. Este ajuste normaliza a média de
    cada canal de cor (R, G, B) para que fiquem próximas entre si,
    deixando a cor da pele mais fiel — sem alterar geometria ou
    textura, portanto sem risco para a identidade facial.

    Args:
        image: Imagem PIL RGB de entrada.

    Returns:
        Imagem PIL RGB com balanço de branco corrigido.
    """
    arr = np.array(image).astype(np.float32)
    channel_means = arr.reshape(-1, 3).mean(axis=0)
    overall_mean = channel_means.mean()

    # Evita divisão por zero em imagens degeneradas (ex.: totalmente pretas).
    safe_means = np.where(channel_means < 1e-3, overall_mean, channel_means)
    gains = overall_mean / safe_means

    # Limita o ganho para não gerar cores artificiais em fotos já neutras.
    gains = np.clip(gains, 0.85, 1.15)

    balanced = arr * gains
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return Image.fromarray(balanced)


def apply_skin_smoothing(
    image: Image.Image, face: Optional[FaceBox], strength: int
) -> Image.Image:
    """Suaviza sutilmente a pele do rosto, sem borrar cabelo/roupa/fundo.

    Usa um filtro bilateral (preserva bordas nítidas, suaviza apenas
    áreas de tom uniforme como a pele) aplicado à imagem inteira, mas
    misturado de volta à imagem original apenas dentro de uma região
    elíptica ao redor do rosto, com borda suavizada (feather). Isso
    reduz brilho/oleosidade e pequenas imperfeições sem "plastificar"
    o rosto nem alterar seus traços — cabelo, roupa e fundo permanecem
    intocados.

    Args:
        image: Imagem PIL RGB (já composta com o novo fundo).
        face: Caixa do rosto detectado, usada para localizar a região a
            suavizar. Se `None`, a função retorna a imagem original
            inalterada (não há como localizar a pele com segurança).
        strength: Intensidade do efeito, de 0 (desligado) a 10 (máximo).

    Returns:
        Imagem PIL RGB com a suavização aplicada (ou a original, se
        `strength == 0` ou nenhum rosto foi fornecido).
    """
    if strength <= 0 or face is None:
        return image

    arr = np.array(image)
    height, width = arr.shape[:2]

    # Filtro bilateral: suaviza tons uniformes mantendo bordas (olhos,
    # sobrancelhas, contorno do nariz/boca) relativamente nítidas.
    d = 9
    sigma = 10 + strength * 4
    smoothed = cv2.bilateralFilter(arr, d=d, sigmaColor=sigma, sigmaSpace=sigma)

    # Máscara elíptica cobrindo a região do rosto, um pouco maior que a
    # caixa detectada, com borda suave para uma transição imperceptível.
    mask = np.zeros((height, width), dtype=np.float32)
    center = face.center
    axis_x = int(face.w * 0.75)
    axis_y = int(face.h * 0.95)
    cv2.ellipse(mask, center, (axis_x, axis_y), 0, 0, 360, 1.0, thickness=-1)
    feather = max(3, min(axis_x, axis_y) // 4) | 1  # garante kernel ímpar
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    mask_3ch = np.stack([mask] * 3, axis=-1)

    blended = arr.astype(np.float32) * (1 - mask_3ch) + smoothed.astype(np.float32) * mask_3ch
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# 6. Redução de ruído (filtro clássico, não generativo)
# --------------------------------------------------------------------------- #

def denoise_image(image: Image.Image, strength: int = 5) -> Image.Image:
    """Reduz ruído usando Non-Local Means Denoising (OpenCV).

    É um filtro determinístico e clássico: suaviza ruído de sensor sem
    "inventar" nenhum detalhe novo, mantendo a identidade intacta.

    Args:
        image: Imagem PIL RGB de entrada.
        strength: Intensidade do filtro (recomendado entre 3 e 10).

    Returns:
        Imagem PIL RGB com ruído reduzido.
    """
    arr_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        arr_bgr, None, h=strength, hColor=strength, templateWindowSize=7, searchWindowSize=21
    )
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(denoised_rgb)


# --------------------------------------------------------------------------- #
# 7. Recorte/enquadramento automático centrado no rosto
# --------------------------------------------------------------------------- #

def auto_crop_to_face(
    image: Image.Image,
    face: FaceBox,
    vertical_padding_ratio: float = 1.8,
    horizontal_padding_ratio: float = 1.4,
) -> Image.Image:
    """Recorta a imagem em um enquadramento tipo "headshot" centrado no rosto.

    Apenas corta a imagem (crop), sem redimensionar ou distorcer os
    traços faciais — a proporção do rosto original é 100% preservada.

    Args:
        image: Imagem PIL RGB completa.
        face: Caixa delimitadora do rosto detectado.
        vertical_padding_ratio: Quanto espaço extra (em múltiplos da
            altura do rosto) deixar acima/abaixo do rosto.
        horizontal_padding_ratio: Quanto espaço extra (em múltiplos da
            largura do rosto) deixar à esquerda/direita.

    Returns:
        Imagem PIL RGB recortada em proporção de retrato (3:4).
    """
    width, height = image.size
    face_cx, face_cy = face.center

    crop_half_h = int(face.h * vertical_padding_ratio)
    crop_half_w = int(face.w * horizontal_padding_ratio)

    # Enquadramento de retrato: um pouco mais alto do que largo (3:4).
    crop_h = crop_half_h * 2
    crop_w = int(crop_h * 3 / 4)
    crop_w = max(crop_w, crop_half_w * 2)

    # Desloca o centro verticalmente para deixar mais espaço para o "peito".
    center_y = face_cy + int(face.h * 0.35)

    left = max(0, face_cx - crop_w // 2)
    right = min(width, face_cx + crop_w // 2)
    top = max(0, center_y - crop_h // 2)
    bottom = min(height, center_y + crop_h // 2)

    # Garante que a caixa não saia dos limites da imagem original.
    if right - left < crop_w:
        if left == 0:
            right = min(width, crop_w)
        else:
            left = max(0, width - crop_w)
    if bottom - top < crop_h:
        if top == 0:
            bottom = min(height, crop_h)
        else:
            top = max(0, height - crop_h)

    return image.crop((left, top, right, bottom))


# --------------------------------------------------------------------------- #
# 8. Redimensionamento final
# --------------------------------------------------------------------------- #

def downscale_for_processing(image: Image.Image, max_dimension: int) -> Image.Image:
    """Reduz a imagem de entrada antes do pipeline pesado, se necessário.

    Fotos muito grandes (ex.: 4000x3000 de uma câmera de celular) deixam
    o `rembg` e os filtros do OpenCV lentos e consomem muita memória —
    um risco real em ambientes com RAM limitada, como o Streamlit Cloud
    gratuito. Esta função aplica o mesmo redimensionamento proporcional
    de `resize_for_output`, mas como proteção de entrada.

    Args:
        image: Imagem PIL RGB original.
        max_dimension: Tamanho máximo permitido para o lado maior.

    Returns:
        Imagem PIL RGB, redimensionada se estava acima do limite.
    """
    return resize_for_output(image, max_dimension)


def resize_for_output(image: Image.Image, max_dimension: int) -> Image.Image:
    """Redimensiona a imagem final mantendo a proporção, sem upscaling forçado.

    Args:
        image: Imagem PIL RGB.
        max_dimension: Tamanho máximo permitido para o lado maior.

    Returns:
        Imagem PIL RGB redimensionada.
    """
    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_dimension:
        return image
    scale = max_dimension / largest_side
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


# --------------------------------------------------------------------------- #
# 9. Integração opcional com API generativa (Replicate/Stability)
# --------------------------------------------------------------------------- #

def refine_background_via_generative_api(
    subject_rgba: Image.Image,
    prompt: str,
) -> Image.Image:
    """Ponto de extensão opcional para inpainting generativo do fundo.

    Esta função é um *stub* documentado: só é chamada se o usuário marcar
    a opção correspondente na UI **e** houver uma chave de API configurada
    em `config.py`/`.env`. Caso contrário, a aplicação usa exclusivamente
    os fundos sintéticos locais (`generate_background`), sem nunca
    depender de rede externa.

    Ao integrar de fato com Replicate ou Stability AI, a chamada deve:
      1. Enviar como máscara de edição APENAS a região de fundo
         (o inverso do canal alpha do sujeito) — nunca a área do rosto/corpo.
      2. Usar parâmetros de baixa intensidade de "denoising strength"
         (ex.: <= 0.3) ou um modo de inpainting estrito, para garantir que
         nenhum pixel do sujeito seja tocado ou redesenhado.
      3. Validar que a imagem de saída tenha exatamente as mesmas dimensões
         e que a máscara do sujeito permaneça idêntica pixel a pixel.

    Args:
        subject_rgba: Imagem RGBA do sujeito com fundo removido.
        prompt: Descrição textual do fundo desejado.

    Returns:
        Imagem PIL RGB com o fundo gerado pela API.

    Raises:
        GenerativeAPIError: Sempre, nesta implementação de referência —
            até que uma chave de API válida e a integração real sejam
            configuradas pelo desenvolvedor.
    """
    if not config.generative_api_available:
        raise GenerativeAPIError(
            "Nenhuma chave de API generativa (Replicate/Stability) foi "
            "configurada. Configure REPLICATE_API_TOKEN ou STABILITY_API_KEY "
            "no arquivo .env para habilitar este recurso."
        )

    # Implementação real fica a cargo do desenvolvedor: aqui deixamos apenas
    # o contrato da função e a validação de segurança acima.
    raise GenerativeAPIError(
        "Integração com API generativa ainda não implementada nesta versão. "
        "Use os fundos de estúdio locais como alternativa."
    )


# --------------------------------------------------------------------------- #
# 10. Pipeline orquestrador
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ProcessingOptions:
    """Agrupa todos os parâmetros ajustáveis do pipeline via UI."""

    background_style: str = "Estúdio cinza neutro"
    brightness: float = 1.08
    contrast: float = 1.08
    denoise_strength: int = 5
    auto_crop: bool = True
    white_balance: bool = True
    skin_smoothing: int = 0
    face_index: int = 0
    rembg_session: object = None


def process_headshot(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """Executa o pipeline completo de headshot profissional.

    Ordem das operações (todas preservando a identidade facial):
        1. Redimensionamento de proteção (evita estourar memória/tempo).
        2. Redução de ruído.
        3. Correção de balanço de branco (opcional).
        4. Detecção de rosto (necessária para recorte e suavização de pele).
        5. Remoção de fundo.
        6. Geração e composição do novo fundo.
        7. Correção de iluminação/contraste.
        8. Suavização de pele localizada no rosto (opcional).
        9. Recorte automático centrado no rosto (opcional).
        10. Redimensionamento final.

    Args:
        image: Imagem PIL RGB original, já validada.
        options: Parâmetros de processamento escolhidos na UI.

    Returns:
        Imagem PIL RGB final, pronta para download.

    Raises:
        NoFaceDetectedError: Se `auto_crop=True` e nenhum rosto for encontrado.
        BackgroundRemovalError: Se a segmentação de fundo falhar.
        ImageProcessingError: Para qualquer outra falha do pipeline (inclui
            estouro de memória em imagens muito grandes).
    """
    try:
        working_image = downscale_for_processing(image, config.MAX_PROCESSING_DIMENSION)
        working_image = denoise_image(working_image, strength=options.denoise_strength)

        if options.white_balance:
            working_image = auto_white_balance(working_image)

        # A detecção de rosto é usada tanto pelo recorte automático quanto
        # pela suavização de pele localizada — só pula se ambos desligados.
        face: Optional[FaceBox] = None
        if options.auto_crop or options.skin_smoothing > 0:
            try:
                face = detect_primary_face(working_image, face_index=options.face_index)
            except NoFaceDetectedError:
                if options.auto_crop:
                    raise
                face = None  # suavização de pele simplesmente não é aplicada

        subject_rgba = remove_background(working_image, session=options.rembg_session)
        background = generate_background(options.background_style, subject_rgba.size)
        composed = composite_with_background(subject_rgba, background)

        lit = enhance_lighting(
            composed, brightness=options.brightness, contrast=options.contrast
        )

        if options.skin_smoothing > 0:
            lit = apply_skin_smoothing(lit, face, strength=options.skin_smoothing)

        if options.auto_crop and face is not None:
            lit = auto_crop_to_face(lit, face)

        return resize_for_output(lit, config.OUTPUT_MAX_DIMENSION)
    except (NoFaceDetectedError, BackgroundRemovalError, GenerativeAPIError):
        raise
    except MemoryError as exc:
        raise ImageProcessingError(
            "A imagem é grande demais para ser processada com a memória "
            "disponível. Tente enviar uma foto com resolução menor."
        ) from exc
