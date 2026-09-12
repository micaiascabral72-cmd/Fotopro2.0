"""
app.py
------
Interface Streamlit da aplicação "Headshot Pro": transforma uma foto
comum em um retrato profissional (estilo LinkedIn/corporativo),
preservando 100% a identidade e os traços faciais do usuário.

Execução local:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging

import streamlit as st
from PIL import Image

from config import config
from image_processor import (
    BackgroundRemovalError,
    GenerativeAPIError,
    ImageProcessingError,
    InvalidImageError,
    NoFaceDetectedError,
    ProcessingOptions,
    process_headshot,
    validate_image,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Headshot Pro — Foto de Perfil Profissional",
    page_icon="📸",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Sidebar — controles de ajuste fino
# --------------------------------------------------------------------------- #

def render_sidebar() -> ProcessingOptions:
    """Renderiza os controles da sidebar e retorna as opções escolhidas."""
    st.sidebar.header("⚙️ Ajustes")

    background_style = st.sidebar.selectbox(
        "Estilo do fundo profissional",
        options=config.BACKGROUND_STYLES,
        index=0,
        help="Escolha o cenário que substituirá o fundo original da sua foto.",
    )

    st.sidebar.subheader("Iluminação")
    brightness = st.sidebar.slider(
        "Brilho", min_value=0.8, max_value=1.4, value=1.08, step=0.02
    )
    contrast = st.sidebar.slider(
        "Contraste", min_value=0.8, max_value=1.4, value=1.08, step=0.02
    )

    st.sidebar.subheader("Nitidez da imagem")
    denoise_strength = st.sidebar.slider(
        "Redução de ruído",
        min_value=0,
        max_value=15,
        value=5,
        help="Valores mais altos suavizam mais o ruído da foto original.",
    )

    st.sidebar.subheader("Enquadramento")
    auto_crop = st.sidebar.checkbox(
        "Recorte automático centrado no rosto", value=True
    )

    if config.generative_api_available:
        st.sidebar.success("🔑 API generativa configurada (recurso experimental).")
    else:
        st.sidebar.caption(
            "💡 Todos os fundos são gerados localmente — nenhuma API externa "
            "é necessária."
        )

    return ProcessingOptions(
        background_style=background_style,
        brightness=brightness,
        contrast=contrast,
        denoise_strength=denoise_strength,
        auto_crop=auto_crop,
    )


# --------------------------------------------------------------------------- #
# Corpo principal
# --------------------------------------------------------------------------- #

def render_header() -> None:
    st.title("📸 Headshot Pro")
    st.markdown(
        "Transforme uma foto comum em um **retrato profissional** para "
        "LinkedIn e uso corporativo — mantendo **100% da sua identidade "
        "e traços faciais** intactos. Apenas iluminação, fundo e "
        "enquadramento são ajustados."
    )
    st.divider()


def render_upload_section() -> st.runtime.uploaded_file_manager.UploadedFile | None:
    uploaded_file = st.file_uploader(
        "Envie sua foto (JPG, JPEG ou PNG)",
        type=list(config.ALLOWED_EXTENSIONS),
        accept_multiple_files=False,
        help=f"Tamanho máximo: {config.MAX_UPLOAD_SIZE_MB} MB.",
    )
    return uploaded_file


def render_result(original_image: Image.Image, processed_image: Image.Image) -> None:
    """Exibe a comparação Antes/Depois e o botão de download."""
    st.subheader("Resultado")
    col_before, col_after = st.columns(2)

    with col_before:
        st.caption("Antes")
        st.image(original_image, use_container_width=True)

    with col_after:
        st.caption("Depois")
        st.image(processed_image, use_container_width=True)

    buffer = io.BytesIO()
    processed_image.save(
        buffer, format="JPEG", quality=config.OUTPUT_JPEG_QUALITY
    )
    buffer.seek(0)

    st.download_button(
        label="⬇️ Baixar foto de perfil profissional (alta resolução)",
        data=buffer,
        file_name="headshot_profissional.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )


def main() -> None:
    render_header()
    options = render_sidebar()
    uploaded_file = render_upload_section()

    if uploaded_file is None:
        st.info("Envie uma foto para começar.")
        return

    file_bytes = uploaded_file.getvalue()

    # --- Validação ---
    try:
        original_image = validate_image(file_bytes, uploaded_file.name)
    except InvalidImageError as exc:
        st.error(f"❌ {exc}")
        return
    except Exception:  # noqa: BLE001
        logger.exception("Erro inesperado ao validar a imagem.")
        st.error(
            "❌ Ocorreu um erro inesperado ao validar sua imagem. "
            "Tente novamente com outro arquivo."
        )
        return

    # --- Processamento ---
    with st.spinner("Gerando sua foto de perfil profissional... isso pode levar alguns segundos."):
        try:
            processed_image = process_headshot(original_image, options)
        except NoFaceDetectedError as exc:
            st.error(f"⚠️ {exc}")
            st.caption(
                "Dica: desative o 'Recorte automático centrado no rosto' na "
                "barra lateral se quiser aplicar apenas fundo e iluminação, "
                "sem depender da detecção facial."
            )
            return
        except BackgroundRemovalError as exc:
            st.error(f"❌ {exc}")
            return
        except GenerativeAPIError as exc:
            st.warning(
                f"⚠️ Recurso de API generativa indisponível ({exc}). "
                "Usando fundo de estúdio local como alternativa."
            )
            return
        except ImageProcessingError as exc:
            st.error(f"❌ Erro ao processar a imagem: {exc}")
            return
        except Exception:  # noqa: BLE001
            logger.exception("Erro inesperado no pipeline de processamento.")
            st.error(
                "❌ Ocorreu um erro inesperado ao processar sua foto. "
                "Tente novamente ou envie outra imagem."
            )
            return

    render_result(original_image, processed_image)


if __name__ == "__main__":
    main()
