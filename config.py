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

# Carrega variáveis de um arquivo .env na raiz do projeto, se existir
# (uso local). Em produção no Streamlit Cloud, isso é ignorado e as
# chaves vêm de `st.secrets` (ver `_get_secret` abaixo).
load_dotenv()


def _get_secret(key: str) -> str | None:
    """Busca uma chave/segredo, priorizando variáveis de ambiente (.env,
    uso local) e caindo para `st.secrets` do Streamlit Cloud como
    alternativa — assim a mesma chave funciona local ou publicada,
    sem precisar duplicar configuração.

    No painel do seu app em https://share.streamlit.io, vá em
    "Settings" → "Secrets" e cole, por exemplo:

        GEMINI_API_KEY = "sua-chave-aqui"

    Salve e reinicie o app — não é necessário criar nenhum arquivo no
    repositório para isso.
    """
    value = os.getenv(key)
    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get(key)
    except Exception:  # noqa: BLE001 - sem Streamlit rodando ou sem secrets.toml
        return None


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

    # --- Limite de processamento interno (proteção de memória/tempo) ---
    # A imagem de entrada é reduzida para no máximo este tamanho ANTES de
    # passar pelo pipeline pesado (denoise, rembg). Isso evita que uma foto
    # muito grande estoure a memória/tempo limitados do Streamlit Cloud.
    MAX_PROCESSING_DIMENSION: int = 1600

    # --- Preview rápido (ajustes ao vivo na sidebar) ---
    PREVIEW_MAX_DIMENSION: int = 380

    # --- Histórico de sessão ---
    MAX_HISTORY_ITEMS: int = 5

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
        default_factory=lambda: _get_secret("REPLICATE_API_TOKEN")
    )
    STABILITY_API_KEY: str | None = field(
        default_factory=lambda: _get_secret("STABILITY_API_KEY")
    )

    # --- IA de sugestão/análise (opcional, apenas texto — nunca edita pixels) ---
    GEMINI_API_KEY: str | None = field(
        default_factory=lambda: _get_secret("GEMINI_API_KEY")
    )
    GEMINI_MODEL_OPTIONS: tuple[str, ...] = (
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro",
    )
    GEMINI_DEFAULT_MODEL: str = "gemini-3.6-flash"

    @property
    def ai_advisor_available(self) -> bool:
        """Indica se a chave do Gemini foi configurada para o assistente de IA."""
        return bool(self.GEMINI_API_KEY)

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
