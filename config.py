"""
config.py
---------
Configurações centrais da aplicação e carregamento seguro de variáveis
de ambiente (chaves de API, limites de upload, parâmetros de pipeline).

Nenhuma chave de API é obrigatória para o funcionamento básico da
aplicação: a substituição de fundo, correção de iluminação, recorte e
redução de ruído funcionam 100% localmente com Pillow/OpenCV/rembg.
As variáveis de API (Replicate/Stability) são apenas para uma
integração opcional e futura via inpainting generativo, sempre
mantendo o rosto original intocado.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Carrega variáveis de um arquivo .env na raiz do projeto, se existir.
load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Agrupa todas as configurações estáticas da aplicação."""

    # --- Limites de upload ---
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png")
    MIN_IMAGE_DIMENSION: int = 200  # px, abaixo disso a imagem é rejeitada

    # --- Saída ---
    OUTPUT_MAX_DIMENSION: int = 2000  # lado maior da imagem final, em px
    OUTPUT_JPEG_QUALITY: int = 95

    # --- Detecção de face (OpenCV Haar Cascade, embutido, sem download) ---
    FACE_CASCADE_SCALE_FACTOR: float = 1.1
    FACE_CASCADE_MIN_NEIGHBORS: int = 6
    FACE_CASCADE_MIN_SIZE: tuple[int, int] = (80, 80)

    # --- Fundos profissionais disponíveis na UI ---
    BACKGROUND_STYLES: tuple[str, ...] = (
        "Estúdio cinza neutro",
        "Estúdio azul corporativo",
        "Escritório desfocado",
        "Branco puro",
        "Gradiente escuro premium",
    )

    # --- Integração opcional com API generativa (desativada por padrão) ---
    REPLICATE_API_TOKEN: str | None = field(
        default_factory=lambda: os.getenv("REPLICATE_API_TOKEN")
    )
    STABILITY_API_KEY: str | None = field(
        default_factory=lambda: os.getenv("STABILITY_API_KEY")
    )

    @property
    def generative_api_available(self) -> bool:
        """Indica se alguma chave de API generativa foi configurada.

        A aplicação nunca depende disso para funcionar; é apenas usada
        para habilitar/desabilitar uma opção extra na sidebar.
        """
        return bool(self.REPLICATE_API_TOKEN or self.STABILITY_API_KEY)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


config = AppConfig()
