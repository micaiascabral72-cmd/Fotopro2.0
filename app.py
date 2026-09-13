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

import base64
import io
import logging
from dataclasses import replace

import streamlit as st
from PIL import Image

import ai_advisor
from config import config
from image_processor import (
    BackgroundRemovalError,
    FaceBox,
    GenerativeAPIError,
    ImageProcessingError,
    InvalidImageError,
    NoFaceDetectedError,
    ProcessingOptions,
    auto_white_balance,
    create_rembg_session,
    detect_faces,
    downscale_for_processing,
    enhance_lighting,
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
# Recursos cacheados — carregados uma única vez por processo do servidor
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Carregando modelo de remoção de fundo...")
def get_rembg_session():
    """Carrega o modelo de segmentação (rembg) uma única vez e reutiliza.

    Sem este cache, o `rembg` recarregaria o modelo do zero a cada foto
    processada, o que é a etapa mais lenta do pipeline.
    """
    return create_rembg_session()


# --------------------------------------------------------------------------- #
# Utilitários de imagem para a UI
# --------------------------------------------------------------------------- #

def _image_to_data_uri(image: Image.Image, quality: int = 85) -> str:
    """Converte uma imagem PIL em uma data URI base64 (para HTML embutido)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_before_after_slider(before: Image.Image, after: Image.Image) -> None:
    """Renderiza um comparador Antes/Depois com slider arrastável.

    Implementado em HTML/CSS/JS puro (via `st.components.v1.html`) para
    não depender de nenhuma biblioteca extra.
    """
    display_width = 480
    ratio = display_width / after.width
    display_height = int(after.height * ratio)

    before_resized = before.resize((display_width, display_height))
    after_resized = after.resize((display_width, display_height))

    before_uri = _image_to_data_uri(before_resized)
    after_uri = _image_to_data_uri(after_resized)

    html = f"""
    <div style="max-width:{display_width}px; margin:0 auto;">
      <div class="hp-compare" style="position:relative; width:100%;
           aspect-ratio:{display_width}/{display_height}; overflow:hidden;
           border-radius:8px; user-select:none;">
        <img src="{after_uri}" style="position:absolute; top:0; left:0;
             width:100%; height:100%; object-fit:cover;" />
        <div class="hp-before-wrap" style="position:absolute; top:0; left:0;
             width:50%; height:100%; overflow:hidden;">
          <img src="{before_uri}" style="position:absolute; top:0; left:0;
               width:{display_width}px; max-width:{display_width}px;
               height:100%; object-fit:cover;" />
        </div>
        <div class="hp-handle" style="position:absolute; top:0; left:50%;
             width:4px; height:100%; background:#fff;
             box-shadow:0 0 6px rgba(0,0,0,0.5); cursor:ew-resize;
             transform:translateX(-2px);">
          <div style="position:absolute; top:50%; left:50%;
               transform:translate(-50%,-50%); width:34px; height:34px;
               background:#fff; border-radius:50%; display:flex;
               align-items:center; justify-content:center;
               box-shadow:0 0 6px rgba(0,0,0,0.5); font-size:14px;">↔</div>
        </div>
        <span style="position:absolute; bottom:8px; left:8px; color:#fff;
              background:rgba(0,0,0,0.55); padding:2px 8px; border-radius:4px;
              font-size:12px;">Antes</span>
        <span style="position:absolute; bottom:8px; right:8px; color:#fff;
              background:rgba(0,0,0,0.55); padding:2px 8px; border-radius:4px;
              font-size:12px;">Depois</span>
      </div>
    </div>
    <script>
      (function() {{
        const container = document.querySelector('.hp-compare');
        const beforeWrap = container.querySelector('.hp-before-wrap');
        const handle = container.querySelector('.hp-handle');
        let dragging = false;

        function setPosition(clientX) {{
          const rect = container.getBoundingClientRect();
          let pct = ((clientX - rect.left) / rect.width) * 100;
          pct = Math.max(0, Math.min(100, pct));
          beforeWrap.style.width = pct + '%';
          handle.style.left = pct + '%';
        }}

        handle.addEventListener('mousedown', () => dragging = true);
        window.addEventListener('mouseup', () => dragging = false);
        window.addEventListener('mousemove', (e) => {{
          if (dragging) setPosition(e.clientX);
        }});
        handle.addEventListener('touchstart', () => dragging = true);
        window.addEventListener('touchend', () => dragging = false);
        window.addEventListener('touchmove', (e) => {{
          if (dragging) setPosition(e.touches[0].clientX);
        }});
        container.addEventListener('click', (e) => setPosition(e.clientX));
      }})();
    </script>
    """
    st.components.v1.html(html, height=display_height + 20)


# --------------------------------------------------------------------------- #
# Sidebar — controles de ajuste fino
# --------------------------------------------------------------------------- #

def render_sidebar(preview_source: Image.Image | None) -> ProcessingOptions:
    """Renderiza os controles da sidebar e retorna as opções escolhidas."""
    st.sidebar.header("⚙️ Ajustes")

    background_style = st.sidebar.selectbox(
        "Estilo do fundo profissional",
        options=config.BACKGROUND_STYLES,
        index=0,
        help="Escolha o cenário que substituirá o fundo original da sua foto.",
    )

    white_balance = st.sidebar.checkbox(
        "Corrigir balanço de branco automaticamente",
        value=True,
        help="Neutraliza tons amarelados/azulados comuns em fotos de celular.",
    )

    st.sidebar.subheader("Iluminação")
    brightness = st.sidebar.slider(
        "Brilho", min_value=0.8, max_value=1.4, value=1.08, step=0.02
    )
    contrast = st.sidebar.slider(
        "Contraste", min_value=0.8, max_value=1.4, value=1.08, step=0.02
    )

    if preview_source is not None:
        preview = auto_white_balance(preview_source) if white_balance else preview_source
        preview = enhance_lighting(preview, brightness=brightness, contrast=contrast)
        st.sidebar.image(
            preview, caption="Preview rápido (sem fundo/recorte)", use_container_width=True
        )

    st.sidebar.subheader("Pele e nitidez")
    denoise_strength = st.sidebar.slider(
        "Redução de ruído",
        min_value=0,
        max_value=15,
        value=5,
        help="Valores mais altos suavizam mais o ruído da foto original.",
    )
    skin_smoothing = st.sidebar.slider(
        "Suavização de pele",
        min_value=0,
        max_value=10,
        value=0,
        help="Suaviza sutilmente a pele do rosto (brilho/oleosidade), "
        "sem borrar cabelo, roupa ou fundo.",
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
        white_balance=white_balance,
        skin_smoothing=skin_smoothing,
    )


# --------------------------------------------------------------------------- #
# Seleção de rosto (quando há mais de uma pessoa na foto)
# --------------------------------------------------------------------------- #

def render_face_picker(image: Image.Image, faces: list[FaceBox]) -> int:
    """Mostra miniaturas dos rostos detectados e retorna o índice escolhido.

    Chamado apenas quando mais de um rosto é detectado na foto.
    """
    st.warning(
        f"👥 Detectei {len(faces)} pessoas nesta foto. Selecione qual é você "
        "para que o recorte e a suavização de pele sejam aplicados corretamente."
    )
    cols = st.columns(len(faces))
    labels = []
    for i, (col, face) in enumerate(zip(cols, faces)):
        pad_x, pad_y = int(face.w * 0.4), int(face.h * 0.4)
        left = max(0, face.x - pad_x)
        top = max(0, face.y - pad_y)
        right = min(image.width, face.x + face.w + pad_x)
        bottom = min(image.height, face.y + face.h + pad_y)
        thumb = image.crop((left, top, right, bottom))
        with col:
            st.image(thumb, caption=f"Pessoa {i + 1}", use_container_width=True)
        labels.append(f"Pessoa {i + 1}")

    choice = st.radio("Quem é você?", options=labels, horizontal=True)
    return labels.index(choice)


# --------------------------------------------------------------------------- #
# Assistente de IA (opcional, apenas sugestões em texto)
# --------------------------------------------------------------------------- #

def render_ai_advisor(original_image: Image.Image) -> None:
    """Renderiza o botão e o resultado do assistente de IA, se configurado.

    Este recurso apenas gera texto com sugestões — nunca edita a imagem.
    """
    with st.expander("🤖 Assistente de IA — sugestões para sua foto (opcional)"):
        if not config.ai_advisor_available:
            st.caption(
                "Configure a variável de ambiente `GEMINI_API_KEY` para "
                "habilitar sugestões de iluminação, enquadramento, roupa e "
                "estilo de fundo geradas por IA."
            )
            return

        model_name = st.selectbox(
            "Modelo do Gemini",
            options=config.GEMINI_MODEL_OPTIONS,
            index=config.GEMINI_MODEL_OPTIONS.index(config.GEMINI_DEFAULT_MODEL),
            help="Modelos 'flash' são mais rápidos e baratos; 'pro' tende a "
            "dar uma análise mais detalhada, porém mais lenta.",
        )

        if st.button("Analisar minha foto com IA"):
            with st.spinner(f"Consultando o Gemini ({model_name})..."):
                try:
                    suggestions = ai_advisor.analyze_photo(original_image, model_name=model_name)
                    st.markdown(suggestions)
                except ai_advisor.AIAdvisorError as exc:
                    st.info(f"Não foi possível gerar sugestões agora: {exc}")


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


def render_upload_section():
    uploaded_file = st.file_uploader(
        "Envie sua foto (JPG, JPEG ou PNG)",
        type=list(config.ALLOWED_EXTENSIONS),
        accept_multiple_files=False,
        help=f"Tamanho máximo: {config.MAX_UPLOAD_SIZE_MB} MB.",
    )
    return uploaded_file


def render_result(
    original_image: Image.Image, processed_image: Image.Image, history_key: str
) -> None:
    """Exibe o comparador Antes/Depois, o download e salva no histórico."""
    st.subheader("Resultado")
    render_before_after_slider(original_image, processed_image)
    st.caption("Arraste o divisor (↔) para comparar antes e depois.")

    buffer = io.BytesIO()
    processed_image.save(buffer, format="JPEG", quality=config.OUTPUT_JPEG_QUALITY)
    buffer.seek(0)

    st.download_button(
        label="⬇️ Baixar foto de perfil profissional (alta resolução)",
        data=buffer,
        file_name="headshot_profissional.jpg",
        mime="image/jpeg",
        use_container_width=True,
        key=f"download_{history_key}",
    )

    _save_to_history(processed_image, label=history_key)


# --------------------------------------------------------------------------- #
# Histórico de sessão
# --------------------------------------------------------------------------- #

def _save_to_history(image: Image.Image, label: str) -> None:
    """Guarda uma versão gerada na sessão, para comparação/download depois."""
    history = st.session_state.setdefault("history", [])

    # Evita duplicar a mesma versão se a página apenas foi re-renderizada.
    if history and history[-1]["label"] == label:
        return

    thumb = image.copy()
    thumb.thumbnail((200, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=config.OUTPUT_JPEG_QUALITY)

    history.append({"label": label, "thumb": thumb, "full_bytes": buffer.getvalue()})
    if len(history) > config.MAX_HISTORY_ITEMS:
        history.pop(0)


def render_history() -> None:
    """Mostra as últimas versões geradas nesta sessão, com download rápido."""
    history = st.session_state.get("history", [])
    if len(history) < 2:
        return  # nada a comparar ainda

    st.divider()
    st.subheader("🕓 Histórico desta sessão")
    st.caption("Compare as versões geradas e baixe qualquer uma delas novamente.")

    cols = st.columns(len(history))
    for col, item in zip(cols, history):
        with col:
            st.image(item["thumb"], use_container_width=True)
            st.download_button(
                "Baixar",
                data=item["full_bytes"],
                file_name=f"headshot_{item['label']}.jpg",
                mime="image/jpeg",
                key=f"history_download_{item['label']}",
                use_container_width=True,
            )


def main() -> None:
    render_header()

    uploaded_file = render_upload_section()

    if uploaded_file is None:
        st.sidebar.header("⚙️ Ajustes")
        st.sidebar.caption("Envie uma foto para ver os ajustes e o preview ao vivo.")
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

    # Reduz a imagem de trabalho para acelerar preview/detecção de rostos.
    working_source = downscale_for_processing(original_image, config.MAX_PROCESSING_DIMENSION)
    preview_thumb = working_source.copy()
    preview_thumb.thumbnail((config.PREVIEW_MAX_DIMENSION, config.PREVIEW_MAX_DIMENSION))

    options = render_sidebar(preview_source=preview_thumb)

    render_ai_advisor(original_image)

    # --- Detecção de múltiplas faces (etapa rápida, antes do pipeline pesado) ---
    faces = detect_faces(working_source)
    face_index = 0
    if len(faces) > 1:
        face_index = render_face_picker(working_source, faces)
    options = replace(options, face_index=face_index, rembg_session=get_rembg_session())

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

    history_key = (
        f"{options.background_style}_{options.brightness}_{options.contrast}_"
        f"{options.skin_smoothing}_{options.auto_crop}"
    ).replace(" ", "-")

    render_result(original_image, processed_image, history_key=history_key)
    render_history()


if __name__ == "__main__":
    main()
