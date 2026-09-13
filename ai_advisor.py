"""
ai_advisor.py
-------------
Assistente de IA **opcional** que analisa a foto enviada e devolve
sugestões em texto (iluminação, enquadramento, roupa, estilo de fundo
recomendado).

Princípio de segurança: este módulo **nunca gera nem edita pixels da
imagem**. Ele só conversa com um modelo multimodal (Gemini) para obter
uma avaliação textual. A imagem final continua sendo produzida
exclusivamente pelo pipeline determinístico de `image_processor.py`,
que é o único responsável por preservar a identidade facial.

Se `GEMINI_API_KEY` não estiver configurada, todas as funções aqui
retornam `None` de forma graciosa — a aplicação principal funciona
100% sem este módulo.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

from config import config

logger = logging.getLogger(__name__)


class AIAdvisorError(Exception):
    """Falha ao consultar o assistente de IA (rede, chave inválida, etc.)."""


_ANALYSIS_PROMPT = """
Você é um fotógrafo profissional especializado em retratos corporativos
(headshots) para LinkedIn. Analise a foto enviada e responda em
português, em no máximo 4 tópicos curtos (bullet points), cobrindo:
- Qualidade da iluminação atual.
- Enquadramento/pose.
- Adequação da roupa para um ambiente corporativo (se visível).
- Qual destes estilos de fundo combinaria melhor com esta pessoa: {styles}.

Seja direto e construtivo. Não descreva a aparência física, identidade,
etnia ou características pessoais da pessoa — foque apenas em aspectos
técnicos da foto (luz, enquadramento, roupa, fundo sugerido).
""".strip()


def _get_client():
    """Importa e configura o SDK do Gemini sob demanda.

    Mantido como import tardio para que a aplicação principal funcione
    normalmente mesmo se `google-generativeai` não estiver instalado.
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise AIAdvisorError(
            "A biblioteca 'google-generativeai' não está instalada."
        ) from exc

    if not config.GEMINI_API_KEY:
        raise AIAdvisorError("GEMINI_API_KEY não configurada no ambiente.")

    genai.configure(api_key=config.GEMINI_API_KEY)
    return genai


def analyze_photo(image: Image.Image) -> str:
    """Envia a foto para o Gemini e retorna sugestões textuais de melhoria.

    Args:
        image: Imagem PIL RGB original (antes do processamento).

    Returns:
        Texto com as sugestões, em markdown simples (bullets).

    Raises:
        AIAdvisorError: Se a chave não estiver configurada, o SDK não
            estiver instalado, ou a chamada à API falhar por qualquer motivo.
    """
    genai = _get_client()

    # Reduz a imagem antes de enviar — economiza banda/tokens e é
    # suficiente para uma avaliação de luz/enquadramento/roupa.
    thumb = image.copy()
    thumb.thumbnail((768, 768))
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=90)

    prompt = _ANALYSIS_PROMPT.format(styles=", ".join(config.BACKGROUND_STYLES))

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": buffer.getvalue()},
            ]
        )
        text = (response.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha ao consultar o assistente de IA: %s", exc)
        raise AIAdvisorError(
            "Não foi possível obter sugestões da IA agora. Tente novamente "
            "mais tarde."
        ) from exc

    if not text:
        raise AIAdvisorError("A IA não retornou nenhuma sugestão para esta foto.")

    return text
