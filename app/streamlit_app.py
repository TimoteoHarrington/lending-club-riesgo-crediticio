"""
Interfaz para analistas de crédito: carga una solicitud y muestra la PD, la
banda, la prima y la decisión. No carga el modelo: todo pasa por la API.

Correr en local (con la API levantada en otra terminal):
    streamlit run app/streamlit_app.py
La dirección de la API se toma de API_URL (variable de entorno o secrets de
Streamlit). Por defecto, http://localhost:8000.
"""
import base64
import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Riesgo de default · Lending Club", page_icon="📊", layout="wide")


def _api_url():
    try:
        url = st.secrets.get("API_URL")
    except Exception:
        url = None
    return (url or os.environ.get("API_URL") or "http://localhost:8000").rstrip("/")


API_URL = _api_url()
TIMEOUT = 90  # el plan gratis de Render tarda ~1 minuto en despertar

VIVIENDAS = {"RENT": "Alquila", "MORTGAGE": "Tiene hipoteca", "OWN": "Propia", "OTHER": "Otra"}
PROPOSITOS = {
    "debt_consolidation": "Consolidar deudas", "credit_card": "Pagar tarjetas",
    "home_improvement": "Refacción de la casa", "other": "Otro", "major_purchase": "Compra grande",
    "medical": "Gastos médicos", "small_business": "Pequeño negocio", "car": "Auto",
    "moving": "Mudanza", "vacation": "Vacaciones", "house": "Casa", "wedding": "Casamiento",
    "renewable_energy": "Energía renovable", "educational": "Educación",
}
DECISION = {
    "aprobar": ("Aprobar", "#1E8E5A", "#E7F5EE"),
    "revisar": ("Revisar", "#B7791F", "#FDF3E1"),
    "rechazar": ("Rechazar", "#C0392B", "#FBEAE8"),
}

PERFILES = {
    "Bajo riesgo": dict(monto=8000, ingreso_anual=120000, dti=8.0, fico=780, antiguedad_crediticia_anios=20.0,
                        antiguedad_laboral_anios=10, vivienda="MORTGAGE", proposito="credit_card", estado="TX",
                        acc_open_past_24mths=2, inq_last_6mths=0, revol_util=20.0, mort_acc=2),
    "Riesgo medio": dict(monto=15000, ingreso_anual=65000, dti=19.0, fico=695, antiguedad_crediticia_anios=12.0,
                         antiguedad_laboral_anios=4, vivienda="RENT", proposito="debt_consolidation", estado="CA",
                         acc_open_past_24mths=5, inq_last_6mths=1),
    "Alto riesgo": dict(monto=30000, ingreso_anual=42000, dti=33.0, fico=665, antiguedad_crediticia_anios=5.0,
                        antiguedad_laboral_anios=None, vivienda="RENT", proposito="small_business", estado="NV",
                        acc_open_past_24mths=12, inq_last_6mths=4, revol_util=90.0, num_tl_op_past_12m=6),
}


# ---------------------------------------------------------------------------
# Llamadas a la API
# ---------------------------------------------------------------------------

def llamar(metodo, ruta, **kwargs):
    """Devuelve (datos, error). Nunca levanta excepciones hacia la interfaz."""
    try:
        r = requests.request(metodo, f"{API_URL}{ruta}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException:
        return None, f"No se pudo conectar con la API en {API_URL}."
    try:
        datos = r.json()
    except ValueError:
        return None, f"La API respondió algo inesperado (código {r.status_code})."
    if r.ok:
        return datos, None
    detalles = datos.get("detalles")
    if detalles:
        return None, "Revisá estos datos:\n" + "\n".join(f"- **{d['campo']}**: {d['mensaje']}" for d in detalles)
    return None, datos.get("error", f"Error {r.status_code}")


@st.cache_data(ttl=300, show_spinner=False)
def info_modelo():
    return llamar("GET", "/modelo")


@st.cache_data(ttl=3600, show_spinner=False)
def opciones():
    return llamar("GET", "/opciones")


def pct(x, dec=1):
    return "–" if x is None else f"{x * 100:.{dec}f}%".replace(".", ",")


def usd(x):
    return f"US$ {x:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------
# Encabezado y estado de la API
# ---------------------------------------------------------------------------

REPO = "https://github.com/lucaskim26/lending-club-riesgo-crediticio"
PRESENTACION = Path(__file__).parent / "presentacion.html"


@st.cache_data(show_spinner=False)
def presentacion_html():
    return PRESENTACION.read_text(encoding="utf-8")


ASSETS = Path(__file__).parent / "assets"


@st.cache_data(show_spinner=False)
def logo(nombre):
    return base64.b64encode((ASSETS / nombre).read_bytes()).decode()


st.markdown(f"""
<div style="display:flex;align-items:center;gap:22px;flex-wrap:wrap;margin:-1.5rem 0 1.2rem">
  <img src="data:image/png;base64,{logo('logo_itba.png')}" alt="ITBA" style="height:64px;border-radius:6px">
  <img src="data:image/png;base64,{logo('logo_lendingclub.png')}" alt="Lending Club" style="height:40px">
  <span style="margin-left:auto;background:#E3EEFC;color:#287AE6;font-weight:700;font-size:14px;
        padding:6px 14px;border-radius:999px">Entrega final · E04</span>
</div>""", unsafe_allow_html=True)

st.title("Evaluador de riesgo crediticio")
st.markdown(
    "**Entrega final de Ciencia de Datos Aplicada (ITBA)** · Gianluca Giannine Lizarraga, Timoteo Harrington y Lucas Kim  \n"
    "Un modelo de machine learning (LightGBM), entrenado con 618 mil préstamos de Lending Club, estima la probabilidad "
    "de que una solicitud de préstamo no se pague. Con eso calcula la **prima de riesgo** y propone una **decisión**: "
    "aprobar, revisar o rechazar.")

PASOS_HTML = "".join(
    f"<div style='flex:1;min-width:200px;background:#F1F5FA;border-radius:10px;padding:14px 18px'>"
    f"<div style='color:#E96732;font-weight:800;font-size:22px'>{n}</div>"
    f"<div style='font-weight:700;color:#172B4D'>{t}</div>"
    f"<div style='color:#5C6F87;font-size:14px'>{d}</div></div>"
    for n, t, d in [
        ("1", "Cargá una solicitud", "Completá el formulario o elegí uno de los ejemplos (bajo, medio o alto riesgo)."),
        ("2", "El modelo la evalúa", "Estima la probabilidad de default (PD) y ubica el préstamo en una de 7 bandas de riesgo."),
        ("3", "Mirá el resultado", "La prima (cuánto se espera perder) y la decisión, explicadas en criollo."),
    ])
st.markdown(f"<div style='display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 14px'>{PASOS_HTML}</div>",
            unsafe_allow_html=True)

c1, c2, c3, _ = st.columns([1, 1, 1, 2])
c1.link_button("Código en GitHub", REPO, use_container_width=True)
c2.link_button("Notebook 04 (prima)", f"{REPO}/blob/main/notebooks/04_pricing.ipynb", use_container_width=True)
c3.link_button("Docs de la API", f"{API_URL}/docs", use_container_width=True)

tab_uno, tab_lote, tab_modelo, tab_pres = st.tabs(
    ["Evaluar una solicitud", "Evaluar un lote (CSV)", "Cómo funciona", "Presentación de la entrega"])

# ---------------------------------------------------------------------------
# Presentación (no depende de la API)
# ---------------------------------------------------------------------------

with tab_pres:
    st.markdown("La presentación de la entrega final: el problema, los datos, el modelo, de la PD a la decisión, "
                "el despliegue y la reflexión.")
    deck = presentacion_html()
    st.components.v1.html(deck, height=720, scrolling=False)
    c1, c2 = st.columns([3, 1])
    c1.caption("Hacé clic en la presentación y usá las flechas (o clic a los lados) para avanzar. "
               "N muestra el guion, O el índice y F la pantalla completa.")
    c2.download_button("Descargar presentación (HTML)", deck.encode("utf-8"), "presentacion_lending_club_e04.html",
                       "text/html", use_container_width=True)

with st.spinner("Conectando con la API (si estaba dormida puede tardar un minuto)..."):
    modelo, error_modelo = info_modelo()
    opc, error_opc = opciones()

with st.sidebar:
    st.subheader("Servicio")
    if error_modelo:
        st.error(error_modelo)
        if st.button("Reintentar"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.success("API conectada")
        st.caption(f"`{API_URL}`")
        st.markdown(f"[Documentación de la API]({API_URL}/docs)")
        if not modelo.get("parametros_completos"):
            st.warning("La API todavía usa la severidad única y no calibra la PD.")
    st.divider()
    st.subheader("Proyecto")
    st.caption("Entrega final (E04) de Ciencia de Datos Aplicada, ITBA 2026. "
               "Datos: préstamos históricos de Lending Club (Kaggle).")
    st.markdown(f"- [Repositorio en GitHub]({REPO})\n"
                f"- [Notebook 04: calibración y prima]({REPO}/blob/main/notebooks/04_pricing.ipynb)\n"
                f"- [Notebook 03: modelado]({REPO}/blob/main/notebooks/03_modelado.ipynb)\n"
                f"- [Notebook 02: datos]({REPO}/blob/main/notebooks/02_recopilacion_preparacion_datos.ipynb)")

if error_modelo or error_opc:
    for tab in (tab_uno, tab_lote, tab_modelo):
        tab.error("No se pudo conectar con la API. Probá de nuevo en un minuto con el botón Reintentar de la barra lateral.")
    st.stop()

# ---------------------------------------------------------------------------
# Una solicitud
# ---------------------------------------------------------------------------

with tab_uno:
    st.markdown("Elegí un ejemplo o completá los datos de la persona que pide el préstamo y tocá **Evaluar solicitud**. "
                "Los datos principales son obligatorios; los del buró de crédito son opcionales.")
    perfil = st.radio("Cargar un ejemplo", ["Personalizado", *PERFILES], horizontal=True,
                      help="Completa el formulario con un perfil de ejemplo. Después se puede cambiar cualquier dato.")
    base = PERFILES.get(perfil, PERFILES["Riesgo medio"])
    clave = perfil.replace(" ", "_")

    with st.form(f"solicitud_{clave}"):
        st.markdown("##### Datos principales")
        c1, c2, c3 = st.columns(3)
        monto = c1.number_input("Monto pedido (USD)", 500, 40000, int(base["monto"]), step=500)
        ingreso = c2.number_input("Ingreso anual (USD)", 1, 10_000_000, int(base["ingreso_anual"]), step=1000)
        dti = c3.number_input("Deudas sobre ingreso, DTI (%)", 0.0, 100.0, float(base["dti"]), step=0.5,
                              help="Pagos mensuales de deudas (sin hipoteca) sobre el ingreso mensual.")
        c1, c2, c3 = st.columns(3)
        fico = c1.number_input("Puntaje FICO", 300, 850, int(base["fico"]), step=5)
        antig_cred = c2.number_input("Años de historial crediticio", 0.0, 80.0,
                                     float(base["antiguedad_crediticia_anios"]), step=0.5)
        opciones_lab = ["No informa", *range(0, 11)]
        lab_actual = base["antiguedad_laboral_anios"]
        antig_lab = c3.selectbox("Años en el empleo actual", opciones_lab,
                                 index=0 if lab_actual is None else opciones_lab.index(lab_actual),
                                 format_func=lambda v: "10 o más" if v == 10 else str(v))
        c1, c2, c3 = st.columns(3)
        vivienda = c1.selectbox("Vivienda", list(VIVIENDAS), list(VIVIENDAS).index(base["vivienda"]),
                                format_func=VIVIENDAS.get)
        proposito = c2.selectbox("Destino del préstamo", opc["propositos"], opc["propositos"].index(base["proposito"]),
                                 format_func=lambda p: PROPOSITOS.get(p, p))
        estado = c3.selectbox("Estado", opc["estados"], opc["estados"].index(base["estado"]))

        with st.expander("Datos del buró de crédito (opcionales)"):
            st.caption("Los que quedan vacíos se completan con el valor típico (mediana) de los préstamos de entrenamiento.")
            buro = {}
            columnas = st.columns(2)
            for i, campo in enumerate(opc["buro"]):
                valor = base.get(campo["campo"])
                buro[campo["campo"]] = columnas[i % 2].number_input(
                    campo["etiqueta"], min_value=float(campo["minimo"]), max_value=float(campo["maximo"]),
                    value=None if valor is None else float(valor), placeholder="Valor típico")

        enviado = st.form_submit_button("Evaluar solicitud", type="primary", use_container_width=True)

    if enviado:
        pedido = dict(monto=monto, ingreso_anual=ingreso, dti=dti, fico=fico,
                      antiguedad_crediticia_anios=antig_cred,
                      antiguedad_laboral_anios=None if antig_lab == "No informa" else antig_lab,
                      vivienda=vivienda, proposito=proposito, estado=estado,
                      **{k: v for k, v in buro.items() if v is not None})
        with st.spinner("Evaluando..."):
            r, error = llamar("POST", "/evaluar", json=pedido)
        if error:
            st.error(error)
        else:
            titulo, color, fondo = DECISION[r["decision"]]
            st.markdown(
                f"<div style='background:{fondo};border-radius:10px;padding:18px 22px;margin:8px 0 16px'>"
                f"<div style='color:{color};font-size:30px;font-weight:700'>{titulo}</div>"
                f"<div style='color:#172B4D'>{r['motivo']}</div></div>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Probabilidad de default", pct(r["pd"]),
                      help="Chance de que el préstamo no se pague en los 36 meses (PD calibrada).")
            m2.metric("Banda de riesgo", f"{r['banda']} de 7")
            m3.metric("Prima de riesgo", pct(r["prima"], 2),
                      help="Pérdida esperada = PD × severidad, como % del monto, por toda la vida del préstamo.")
            m4.metric("Pérdida esperada", usd(r["perdida_esperada_usd"]))

            # Franja de bandas
            celdas = []
            for b in modelo["bandas"]:
                activa = b["banda"] == r["banda"]
                c = DECISION[b["decision"]][1]
                celdas.append(
                    f"<div style='flex:1;text-align:center;padding:10px 4px;border-radius:6px;"
                    f"background:{c if activa else '#F1F5FA'};color:{'#fff' if activa else '#5C6F87'};"
                    f"font-weight:{700 if activa else 400}'>Banda {b['banda']}<br>"
                    f"<span style='font-size:12px'>{DECISION[b['decision']][0]}</span></div>")
            st.markdown("<div style='display:flex;gap:6px;margin:6px 0 14px'>" + "".join(celdas) + "</div>",
                        unsafe_allow_html=True)

            de_cada_100 = round(r["pd"] * 100)
            texto = (f"**Cómo se lee:** de cada 100 préstamos como este, se espera que unos **{de_cada_100}** no se paguen. "
                     f"Si no se paga, en esta banda se pierde en promedio el **{pct(r['severidad'], 0)}** del monto. "
                     f"Por eso la prima es {pct(r['pd'])} × {pct(r['severidad'], 0)} = **{pct(r['prima'], 2)}** "
                     f"del monto, unos **{usd(r['perdida_esperada_usd'])}** en toda la vida del préstamo.")
            if r.get("default_historico_banda") is not None:
                texto += f" En test, los préstamos de la banda {r['banda']} tuvieron {pct(r['default_historico_banda'])} de default."
            st.markdown(texto)
            if r["variables_completadas"]:
                st.caption(f"Se completaron con valores típicos {len(r['variables_completadas'])} datos del buró que no se informaron.")
            with st.expander("Ver el pedido y la respuesta de la API"):
                st.markdown(f"`POST {API_URL}/evaluar`")
                c1, c2 = st.columns(2)
                c1.json(pedido)
                c2.json(r)

# ---------------------------------------------------------------------------
# Lote
# ---------------------------------------------------------------------------

with tab_lote:
    st.markdown("Subí un CSV con una solicitud por fila (hasta 1000). Las columnas son las mismas que en el formulario; "
                "las del buró pueden quedar vacías.")
    plantilla = pd.DataFrame([dict(p) for p in PERFILES.values()])
    st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_solicitudes.csv", "text/csv")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            tabla = pd.read_csv(archivo)
        except Exception:
            st.error("No se pudo leer el archivo. Tiene que ser un CSV separado por comas.")
            tabla = None
        if tabla is not None:
            st.caption(f"{len(tabla)} solicitudes leídas.")
            if st.button("Evaluar el lote", type="primary"):
                registros = json.loads(tabla.to_json(orient="records"))
                registros = [{k: v for k, v in fila.items() if v is not None} for fila in registros]
                with st.spinner("Evaluando..."):
                    r, error = llamar("POST", "/evaluar/lote", json={"solicitudes": registros})
                if error:
                    st.error(error.replace("solicitudes.", "fila "))
                else:
                    res = pd.DataFrame(r["resultados"])[
                        ["decision", "pd", "banda", "severidad", "prima", "perdida_esperada_usd", "pd_modelo"]]
                    salida = pd.concat([res, tabla.reset_index(drop=True)], axis=1)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Aprobar", r["resumen"]["aprobar"])
                    c2.metric("Revisar", r["resumen"]["revisar"])
                    c3.metric("Rechazar", r["resumen"]["rechazar"])
                    st.dataframe(salida, use_container_width=True, hide_index=True)
                    buffer = io.StringIO()
                    salida.to_csv(buffer, index=False)
                    st.download_button("Descargar resultados", buffer.getvalue().encode("utf-8"),
                                       "resultados.csv", "text/csv")

# ---------------------------------------------------------------------------
# Cómo funciona
# ---------------------------------------------------------------------------

with tab_modelo:
    st.markdown("#### El problema")
    st.markdown(
        "Lending Club presta plata entre personas. De los préstamos a 36 meses, cerca del **15% no se paga**, y cuando "
        "eso pasa se pierde en promedio la mitad de lo prestado. La pregunta es: **¿qué tan probable es que esta "
        "solicitud no se pague, cuánto habría que cobrarle por ese riesgo y conviene aprobarla?**")

    st.markdown("#### Qué hace el sistema con cada solicitud")
    st.markdown(
        "1. **Arma los datos.** Toma los datos principales y, si están, los del buró. Lo que falta se completa con "
        "el valor típico (mediana) de los préstamos de entrenamiento.\n"
        "2. **Estima la PD.** El modelo LightGBM calcula la probabilidad de default y la **calibra** (ajuste de "
        "intercepto) para que, en promedio, coincida con el default real.\n"
        "3. **Asigna la banda.** La PD ubica al préstamo en una de **7 bandas de riesgo**.\n"
        "4. **Calcula la prima.** **Prima = PD × severidad de la banda**: la pérdida esperada de toda la vida del "
        "préstamo, como % del monto.\n"
        "5. **Decide.** Bandas 1 a 4: aprobar. Banda 5: revisar. Bandas 6 y 7: rechazar.")

    st.markdown("#### Qué tan bien funciona (préstamos de test que el modelo nunca vio)")
    met = modelo.get("metricas_test", {})
    cal = modelo.get("metricas_test_calibrado") or met
    if met:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{met['roc_auc']:.3f}".replace(".", ","),
                  help="Qué tan bien ordena el riesgo: 0,5 es azar y 1 es perfecto. La nota de Lending Club logra 0,683.")
        c2.metric("Nota de Lending Club", "0,683", help="ROC-AUC de sub_grade, la calificación que usaba Lending Club, en los mismos préstamos.")
        c3.metric("PD media (calibrada)", pct(cal["pd_media"]))
        c4.metric("Default real", pct(cal["default_observado"]))
        st.caption("La prima cubre el 99% de la pérdida real de test. Con la regla de decisión, la pérdida de los "
                   "préstamos aprobados baja de 7,8% a 6,9% del monto.")

    st.markdown("#### Las 7 bandas de riesgo")
    bandas = pd.DataFrame(modelo["bandas"])
    vista = pd.DataFrame({
        "Banda": bandas["banda"],
        "Rango de PD": [f"{pct(d)} a {pct(h)}" for d, h in zip(bandas["pd_desde"], bandas["pd_hasta"])],
        "Default en test": bandas["default_historico"].map(pct),
        "Severidad": bandas["severidad"].map(lambda s: pct(s, 1)),
        "Decisión": bandas["decision"].map(lambda d: DECISION[d][0])})
    st.dataframe(vista, use_container_width=True, hide_index=True)

    with st.expander("Glosario"):
        st.markdown(
            "- **PD (probabilidad de default):** la chance de que el préstamo no se devuelva en los 36 meses.\n"
            "- **Calibración:** ajuste para que la PD promedio coincida con el default real, sin cambiar el orden.\n"
            "- **Banda de riesgo:** grupo de préstamos con PD parecida, de 1 (más seguro) a 7 (más riesgoso).\n"
            "- **Severidad:** qué parte del monto se pierde si el préstamo no se paga.\n"
            "- **Prima de riesgo:** PD × severidad. Lo que hay que cobrar para cubrir la pérdida esperada.\n"
            "- **ROC-AUC:** qué tan bien ordena el modelo a los que pagan y a los que no. 0,5 es azar y 1 es perfecto.\n"
            "- **API:** el servicio que recibe los datos de una solicitud y devuelve el resultado. Esta página la usa por detrás.")
    with st.expander("Límites"):
        st.markdown(
            "- La prima es la pérdida esperada de toda la vida del préstamo, no una tasa anual.\n"
            "- El modelo aprendió de préstamos que Lending Club ya había aprobado: no sabe cómo se comportan los rechazados.\n"
            "- Si cambia el ciclo económico, hay que volver a calibrar con datos recientes.\n"
            "- Cuando faltan datos del buró se usan valores típicos: el resultado es razonable, pero menos preciso.")
    st.markdown("#### Las cuatro entregas")
    st.markdown(
        "| Entrega | Qué hicimos |\n|---|---|\n"
        "| E01 · Propuesta | Problema, objetivos y KPIs |\n"
        "| E02 · Datos | Limpieza, leakage, split temporal y variables (618 mil préstamos, 54 variables) |\n"
        "| E03 · Modelado | Logística, Random Forest y LightGBM con validación por año. Ganó LightGBM |\n"
        f"| **E04 · Despliegue** | Calibración, prima, regla de decisión, [API]({API_URL}/docs) y esta página |")
