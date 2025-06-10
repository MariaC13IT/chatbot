import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import re

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
NOMBRE_BOT = "Tutor IA"

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)
embedding_model = get_embedding_model()

PDFS_FOLDER = "pdfs"
pdf_files = [f for f in os.listdir(PDFS_FOLDER) if f.endswith(".pdf")]
if not pdf_files:
    st.warning("No hay PDFs en la carpeta 'pdfs'. Añade al menos uno para empezar.")
    st.stop()

def limpia_lineas(pdf_text):
    pdf_text = re.sub(r'(\w)-\n(\w)', r'\1\2', pdf_text)
    pdf_text = re.sub(r'(?<![.:;?!\d•\-◦])\n(?!\n)', ' ', pdf_text)
    pdf_text = re.sub(r'\n{2,}', '\n\n', pdf_text)
    pdf_text = re.sub(r'[ \t]+\n', '\n', pdf_text)
    pdf_text = re.sub(r'\n[ \t]+', '\n', pdf_text)
    return pdf_text

@st.cache_resource(show_spinner="Procesando materiales PDF...")
def carga_todos_los_pdfs(pdf_files):
    all_chunks, origen, apartados = [], [], []
    for fname in pdf_files:
        pdf_path = os.path.join(PDFS_FOLDER, fname)
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[:50]:
            t = page.extract_text()
            if t: text += t + "\n"
        text = limpia_lineas(text)
        secciones = re.split(r'(?=(\d+\.\d+\.\s+.+\n|¿.+?\?))', text)
        for i in range(1, len(secciones), 2):
            titulo = secciones[i].strip()
            contenido = secciones[i+1].strip() if (i+1)<len(secciones) else ""
            bloque = f"{titulo}\n{contenido}"
            if len(bloque) > 60:
                all_chunks.append(bloque)
                origen.append(fname)
                if titulo:
                    apartados.append(titulo)
                else:
                    apartados.append(bloque[:35])
        if not all_chunks:
            parrafos = text.split('\n\n')
            for p in parrafos:
                if len(p) > 40:
                    all_chunks.append(p.strip())
                    origen.append(fname)
                    apartados.append(p.strip()[:35])
    return all_chunks, origen, apartados

chunks, chunks_origen, chunks_apartado = carga_todos_los_pdfs(pdf_files)
embeddings = embedding_model.encode(chunks)

def resalta_palabras(texto, palabras):
    for palabra in set(palabras):
        if len(palabra) > 3:
            texto = re.sub(f"({re.escape(palabra)})", r"<mark>\1</mark>", texto, flags=re.IGNORECASE)
    return texto

def mejor_frase(chunk, pregunta):
    frases = re.split(r'(?<=[.?!])\s+', chunk)
    frases_filtradas = [f for f in frases if len(f) > 40 and not pregunta.strip().lower() in f.lower()]
    if frases_filtradas:
        return max(frases_filtradas, key=len)
    elif frases:
        return frases[0]
    else:
        return chunk[:200] + "..."

# Estado inicial
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'to_answer' not in st.session_state:
    st.session_state.to_answer = ""
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

st.title("🤖 Asistente de Estudio")
st.markdown(
    "<div style='background:#eaf6ff; padding:18px 22px; border-radius:16px; font-size:1.17em; margin-bottom:18px;'>"
    "👋 <b>¡Hola! Soy tu asistente de estudio.</b> Puedes preguntar sobre todos tus materiales a la vez."
    "</div>", unsafe_allow_html=True
)

# Mostrar historial de chat
for item in st.session_state.chat_history:
    # Usuario derecha
    st.markdown(
        f"<div style='display:flex;justify-content:flex-end;margin-bottom:8px;'>"
        f"<div style='background:#e6f4ea;padding:13px 17px 10px 17px;"
        f"border-radius:16px 16px 3px 16px;max-width:72%;margin-left:8px;'>"
        f"<b style='color:#146b6b;'>Tú:</b><br>{item['pregunta']}</div></div>",
        unsafe_allow_html=True)
    # IA izquierda
    st.markdown(
        f"<div style='display:flex;justify-content:flex-start;margin-bottom:8px;'>"
        f"<div style='background:#f1edff;padding:13px 17px 10px 17px;"
        f"border-radius:16px 16px 16px 3px;max-width:72%;margin-right:8px;'>"
        f"<b style='color:#783da6;'>{NOMBRE_BOT}:</b><br>"
        f"{item['respuesta_directa']}"
        f"<div style='font-size:0.92em; color:#888; margin-top:7px;'>"
        f"Si tienes dudas acude a este contenido <b>{item['pdf_sin_ext']}</b> en el apartado <b>{item['apartado']}</b>.</div>"
        f"</div></div>",
        unsafe_allow_html=True
    )

# INPUT de usuario con manejo correcto del estado
def handle_submit():
    if st.session_state.user_input.strip():
        st.session_state.to_answer = st.session_state.user_input.strip()
        st.session_state.input_text = st.session_state.user_input  # Guardamos el input actual
        st.session_state.user_input = ""  # Limpiamos el campo de texto

# Usamos un formulario para manejar mejor el submit
with st.form("chat_form"):
    user_input = st.text_input(
        "Pregunta lo que quieras 👇",
        value=st.session_state.get('user_input', ''),
        key="user_input",
        placeholder="Ejemplo: ¿Por qué es grave la prevaricación administrativa?"
    )
    submitted = st.form_submit_button("Preguntar", on_click=handle_submit)

# PROCESA la respuesta si hay algo pendiente
if st.session_state.to_answer:
    question = st.session_state.to_answer
    question_embedding = embedding_model.encode([question])[0]
    similarities = np.dot(embeddings, question_embedding)
    idx = int(np.argmax(similarities))
    max_sim = similarities[idx]
    SIMILARIDAD_MINIMA = 0.65

    if max_sim < SIMILARIDAD_MINIMA:
        st.session_state.chat_history.append({
            "pregunta": question,
            "respuesta_directa": "<i>Pregunta fuera del contenido didáctico. Solo puedo responder sobre el material proporcionado.</i>",
            "pdf_sin_ext": "-",
            "apartado": "-",
            "similaridad": float(max_sim)
        })
    else:
        best_chunk = chunks[idx]
        respuesta_directa = mejor_frase(best_chunk, question)
        respuesta_directa = resalta_palabras(respuesta_directa, question.split())
        nombre_pdf = os.path.splitext(chunks_origen[idx])[0]
        apartado = chunks_apartado[idx][:45] + ("..." if len(chunks_apartado[idx]) > 45 else "")
        st.session_state.chat_history.append({
            "pregunta": question,
            "respuesta_directa": respuesta_directa,
            "pdf_sin_ext": nombre_pdf,
            "apartado": apartado,
            "similaridad": float(max_sim)
        })
    st.session_state.to_answer = ""  # LIMPIA después

st.markdown("""
---
<sub>
Si tienes dudas acude al material original.
</sub>
""", unsafe_allow_html=True)