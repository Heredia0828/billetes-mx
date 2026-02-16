# app.py — Detector de seguridad en billetes mexicanos 💵
# Proyecto de visión por computadora con Streamlit + OpenCV

import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison
from gtts import gTTS
import matplotlib.pyplot as plt

# Config de la página
st.set_page_config(page_title="Seguridad en Billetes MXN", page_icon="💵", layout="wide")

# Inicializar el state pa la caja registradora
if "dinero_acumulado" not in st.session_state:
    st.session_state.dinero_acumulado = 0
if "conteo_billetes" not in st.session_state:
    st.session_state.conteo_billetes = 0

# CSS pa que se vea chido el diseño oscuro
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); }
section[data-testid="stSidebar"] { background: rgba(15,32,39,0.92); border-right: 1px solid rgba(0,230,118,0.15); }
.analysis-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(0,230,118,0.18); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; backdrop-filter: blur(12px); }
.result-badge { display: inline-block; background: linear-gradient(135deg,#00e676,#00c853); color: #0f2027; font-weight: 700; padding: 0.45rem 1.1rem; border-radius: 24px; font-size: 0.92rem; margin-top: 0.5rem; }
.main-header { text-align: center; padding: 1.5rem 0 0.5rem; }
.main-header h1 { background: linear-gradient(90deg,#00e676,#00bcd4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.4rem; font-weight: 800; margin-bottom: 0.15rem; }
.main-header p { color: rgba(255,255,255,0.55); font-size: 1.05rem; }
.caja-card { background: linear-gradient(135deg,rgba(0,230,118,0.10),rgba(0,188,212,0.10)); border: 1px solid rgba(0,230,118,0.30); border-radius: 16px; padding: 1.2rem 1.5rem; text-align: center; margin-bottom: 1rem; }
.caja-card .monto { font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg,#00e676,#69f0ae); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.caja-card .label { color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>💵 Seguridad en Billetes Mexicanos</h1>
    <p>Análisis de elementos de seguridad mediante visión por computadora</p>
</div>
""", unsafe_allow_html=True)

# Caja registradora arriba
col_caja1, col_caja2, col_caja3 = st.columns([2, 1, 1])
with col_caja1:
    st.markdown(f"""
    <div class="caja-card">
        <div class="label">💰 Caja Registradora – Dinero Acumulado</div>
        <div class="monto">${st.session_state.dinero_acumulado:,.2f} MXN</div>
    </div>
    """, unsafe_allow_html=True)
with col_caja2:
    st.metric("🧾 Billetes verificados", st.session_state.conteo_billetes)
with col_caja3:
    if st.button("🗑️ Resetear Caja", use_container_width=True):
        st.session_state.dinero_acumulado = 0
        st.session_state.conteo_billetes = 0
        st.rerun()


# ======================== FUNCIONES DE ANÁLISIS ========================
# Cada una recibe img BGR y regresa (resultado, descripción, pasos_intermedios)

def analizar_marca_de_agua(img):
    # Pasamos a grises y le metemos CLAHE pa realzar detalles ocultos
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    resultado = clahe.apply(gris)
    pasos = [
        ("Paso 1 – Escala de grises", gris.copy(),
         "Se quita el color y queda solo la luminancia del billete."),
        ("Paso 2 – CLAHE (ecualización adaptativa)", resultado.copy(),
         "CLAHE ecualiza el histograma por bloques de 8×8, realza detalles sin quemar las zonas claras."),
    ]
    desc = "**Marca de agua – CLAHE aplicado.** Se realzaron variaciones sutiles de luminosidad que podrían ser la marca de agua."
    return resultado, desc, pasos


def detectar_hilo_de_seguridad(img):
    # Canny pa bordes + dilatación vertical pa resaltar el hilo
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilatado = cv2.dilate(bordes, kernel, iterations=2)
    pasos = [
        ("Paso 1 – Escala de grises", gris.copy(),
         "Canal único pa preparar la detección de bordes."),
        ("Paso 2 – Bordes Canny", bordes.copy(),
         "Canny detecta bordes con doble umbral (50/150), saca un mapa binario."),
        ("Paso 3 – Dilatación vertical 1×5", dilatado.copy(),
         "Un kernel vertical conecta bordes rotos y resalta líneas como el hilo de seguridad."),
    ]
    desc = "**Hilo de seguridad – Canny + Dilatación.** Se resaltaron líneas verticales que podrían ser el hilo incrustado."
    return dilatado, desc, pasos


def analizar_microimpresion(img):
    # Laplaciano pa detectar zonas con cambios bruscos de intensidad
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gris, cv2.CV_64F)
    resultado = cv2.convertScaleAbs(lap)
    pasos = [
        ("Paso 1 – Escala de grises", gris.copy(),
         "Canal de luminancia como entrada al Laplaciano."),
        ("Paso 2 – Laplaciano (2ª derivada)", resultado.copy(),
         "Segunda derivada espacial, los valores altos son donde hay microtextos."),
    ]
    desc = "**Microimpresión – Filtro Laplaciano.** Se resaltaron zonas de alta frecuencia donde están las microimpresiones."
    return resultado, desc, pasos


def detectar_tinta_uv(img):
    # Sacamos el canal azul y le ponemos un colormap pa simular UV
    canal_b = img[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    realzado = clahe.apply(canal_b)
    resultado = cv2.applyColorMap(realzado, cv2.COLORMAP_JET)
    pasos = [
        ("Paso 1 – Canal azul extraído", canal_b.copy(),
         "Aislamos el azul porque las tintas UV fluorescen cerca de esa longitud de onda."),
        ("Paso 2 – CLAHE en canal azul", realzado.copy(),
         "Mejoramos el contraste del canal azul aislado."),
        ("Paso 3 – Mapa de calor JET", resultado.copy(),
         "Colormap JET convierte intensidad en gradiente de colores, simula la fluorescencia UV."),
    ]
    desc = "**Tinta UV – Canal azul + Mapa de calor.** Se simuló la fluorescencia de tintas UV del billete."
    return resultado, desc, pasos


def analizar_relieve(img):
    # Emboss pa simular el relieve (intaglio) del billete
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
    resultado = cv2.filter2D(gris, -1, kernel) + 128
    pasos = [
        ("Paso 1 – Escala de grises", gris.copy(),
         "Canal único pa la convolución 2D."),
        ("Paso 2 – Filtro Emboss + offset 128", resultado.copy(),
         "Kernel asimétrico que simula luz lateral, resalta bordes en una dirección."),
    ]
    desc = "**Relieve (Intaglio) – Filtro Emboss.** Se visualizaron zonas con impresión en relieve."
    return resultado, desc, pasos


# ======================== FUNCIONES AUXILIARES ========================

def _to_rgb(img):
    # Convierte gray o BGR a RGB pa mostrar en streamlit
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _to_pil(img):
    return Image.fromarray(_to_rgb(img))

def generar_histograma(img_gray):
    # Histograma bonito con fondo oscuro
    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#0f2027")
    ax.set_facecolor("#0f2027")
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    ax.fill_between(range(256), hist.flatten(), color="#00e676", alpha=0.6)
    ax.plot(hist, color="#69f0ae", linewidth=1)
    ax.set_xlim([0, 256])
    ax.tick_params(colors="white", labelsize=7)
    ax.set_ylabel("Píxeles", color="white", fontsize=8)
    ax.set_xlabel("Intensidad", color="white", fontsize=8)
    ax.set_title("Histograma de intensidades", color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.2))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0f2027", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf

def generar_audio(valor):
    # gTTS pa que diga "billete verificado" (necesita internet)
    tts = gTTS(text=f"Billete de {valor} pesos verificado con éxito.", lang="es", tld="com.mx")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf


# ======================== CATÁLOGO DE ANÁLISIS ========================

ANALISIS = {
    "🔍 Marca de agua": {"fn": analizar_marca_de_agua, "icono": "🔍", "corto": "Marca de agua",
        "info": "Realza luminosidad (CLAHE) pa evidenciar la marca de agua."},
    "🧵 Hilo de seguridad": {"fn": detectar_hilo_de_seguridad, "icono": "🧵", "corto": "Hilo de seguridad",
        "info": "Bordes Canny + dilatación vertical pa resaltar el hilo."},
    "🔬 Microimpresión": {"fn": analizar_microimpresion, "icono": "🔬", "corto": "Microimpresión",
        "info": "Laplaciano pa detectar zonas de alta frecuencia (microtextos)."},
    "💜 Tinta UV": {"fn": detectar_tinta_uv, "icono": "💜", "corto": "Tinta UV",
        "info": "Canal azul + mapa de calor pa simular fluorescencia UV."},
    "✋ Relieve (Intaglio)": {"fn": analizar_relieve, "icono": "✋", "corto": "Relieve",
        "info": "Filtro emboss pa visualizar la impresión en relieve."},
}

DENOMINACIONES = [20, 50, 100, 200, 500, 1000]


# ======================== SIDEBAR ========================

with st.sidebar:
    st.markdown("### 📂 Cargar imagen del billete")

    # Elegir si sube archivo o usa la cámara
    fuente = st.radio("¿Cómo capturas la imagen?", ["📁 Subir archivo", "📷 Usar cámara"], horizontal=True)

    archivo_imagen = None
    foto_camara = None
    if fuente == "📁 Subir archivo":
        archivo_imagen = st.file_uploader("Sube tu imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])
    else:
        foto_camara = st.camera_input("📸 Toma una foto del billete")

    fuente_imagen = archivo_imagen or foto_camara

    st.divider()

    # Denominación del billete (manual, no es automática)
    st.markdown("### 💲 Denominación")
    denominacion = st.selectbox("Valor del billete:", DENOMINACIONES, format_func=lambda x: f"${x:,} MXN", index=0)

    st.divider()

    # Qué filtro aplicar
    st.markdown("### 🛡️ Elemento de seguridad")
    seleccion = st.radio("Análisis:", list(ANALISIS.keys()), index=0)
    info = ANALISIS[seleccion]
    st.info(f"{info['icono']}  {info['info']}")

    st.divider()
    ejecutar = st.button("▶️  Ejecutar análisis", use_container_width=True, type="primary")

    st.divider()
    modo_inspector = st.checkbox("🕵️ Activar Modo Inspector",
        help="Muestra los pasos intermedios del algoritmo y los histogramas.")


# ======================== CONTENIDO PRINCIPAL ========================

if fuente_imagen is not None:
    # Cargar imagen con manejo de errores
    try:
        imagen_pil = Image.open(fuente_imagen).convert("RGB")
        imagen_np = np.array(imagen_pil)
        if imagen_np.size == 0:
            raise ValueError("La imagen está vacía.")
        if imagen_np.shape[0] < 10 or imagen_np.shape[1] < 10:
            raise ValueError("La imagen es muy pequeña (mín 10×10 px).")
        imagen_bgr = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"⚠️ **No se pudo procesar la imagen.** Sube una imagen válida con buena iluminación.\n\nError: `{e}`")
        st.stop()

    # Mostrar la original
    st.markdown('<div class="analysis-card"><h4>📸 Imagen original</h4></div>', unsafe_allow_html=True)
    st.image(imagen_pil, use_container_width=True)

    # Si le dan al botón, corremos el análisis
    if ejecutar:
        st.divider()
        st.markdown(f'<div class="analysis-card"><h4>{info["icono"]} Resultado: {seleccion}</h4></div>', unsafe_allow_html=True)

        try:
            with st.spinner("Procesando imagen…"):
                img_resultado, descripcion, pasos = info["fn"](imagen_bgr)
        except Exception as e:
            st.error(f"⚠️ **Error en el procesamiento.** Intenta con otra imagen.\n\nError: `{e}`")
            st.stop()

        # Slider de comparación original vs procesada
        image_comparison(
            img1=imagen_pil, img2=_to_pil(img_resultado),
            label1="Original", label2=f"Análisis: {info['corto']}",
            width=700, starting_position=50, show_labels=True, make_responsive=True,
        )

        # Resultados
        st.markdown("---")
        st.markdown(descripcion)
        st.markdown('<span class="result-badge">✅ Análisis completado con éxito</span>', unsafe_allow_html=True)

        # Métricas de la imagen
        c1, c2, c3 = st.columns(3)
        c1.metric("Ancho (px)", img_resultado.shape[1])
        c2.metric("Alto (px)", img_resultado.shape[0])
        c3.metric("Canales", 1 if len(img_resultado.shape) == 2 else img_resultado.shape[2])

        # Sumar a la caja registradora
        st.session_state.dinero_acumulado += denominacion
        st.session_state.conteo_billetes += 1
        st.success(f"💰 +${denominacion:,} MXN → Total: **${st.session_state.dinero_acumulado:,.2f} MXN** ({st.session_state.conteo_billetes} billetes)")

        # Audio de confirmación
        st.markdown("---")
        st.markdown("#### 🔊 Confirmación por audio")
        try:
            st.audio(generar_audio(denominacion), format="audio/mp3")
        except Exception:
            st.warning("🔇 No se generó el audio (se necesita internet pa gTTS).")

        # Modo inspector: pasos intermedios + histogramas
        if modo_inspector:
            st.markdown("---")
            with st.expander("🕵️ Modo Inspector – Pasos del algoritmo", expanded=True):
                st.markdown("> Aquí se ven los pasos intermedios de lo que hizo el filtro.")
                for i, (titulo, img_paso, explicacion) in enumerate(pasos):
                    st.markdown(f"##### {titulo}")
                    st.markdown(explicacion)
                    st.image(_to_pil(img_paso), use_container_width=True)
                    if len(img_paso.shape) == 2:
                        st.image(generar_histograma(img_paso), caption=f"Histograma – {titulo}", use_container_width=True)
                    if i < len(pasos) - 1:
                        st.markdown("---")
else:
    # No hay imagen, mostramos mensaje de bienvenida
    st.markdown("""
    <div style="text-align:center; padding:4rem 1rem;">
        <p style="font-size:4rem; margin-bottom:0.5rem;">💵</p>
        <h3 style="color:rgba(255,255,255,0.7);">Carga una imagen de un billete mexicano</h3>
        <p style="color:rgba(255,255,255,0.4);">Usa el panel izquierdo pa seleccionar imagen y tipo de análisis.</p>
    </div>
    """, unsafe_allow_html=True)
