"""
Interfaz para analistas de crédito: carga una solicitud y muestra la PD, la
banda, la prima y la decisión. No carga el modelo: todo pasa por la API.

Correr en local (con la API levantada en otra terminal):
    streamlit run app/streamlit_app.py
La dirección de la API se toma de API_URL (variable de entorno o secrets de
Streamlit). Por defecto, http://localhost:8000.
"""
import io
import json
import os

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

st.title("Evaluador de riesgo crediticio")
st.caption("Probabilidad de default, banda de riesgo, prima y decisión para préstamos personales a 36 meses. "
           "Modelo LightGBM entrenado con datos de Lending Club (2007-2015).")

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

if error_modelo or error_opc:
    st.stop()

tab_uno, tab_lote, tab_modelo = st.tabs(["Evaluar una solicitud", "Evaluar un lote (CSV)", "Cómo funciona"])

# ---------------------------------------------------------------------------
# Una solicitud
# ---------------------------------------------------------------------------

with tab_uno:
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
    st.markdown(
        "1. **Solicitud:** los datos principales y, si se tienen, los del buró. Lo que falta se completa con la mediana de train.\n"
        "2. **PD:** el modelo LightGBM estima la probabilidad de default y se calibra con un ajuste de intercepto, "
        "para que el promedio coincida con el default real.\n"
        "3. **Banda:** la PD ubica al préstamo en una de 7 bandas de riesgo.\n"
        "4. **Prima:** PD × severidad de la banda (cuánto se pierde del monto si no paga).\n"
        "5. **Decisión:** según la banda, se aprueba, se manda a revisión o se rechaza.")
    met = modelo.get("metricas_test", {})
    cal = modelo.get("metricas_test_calibrado") or met
    if met:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC en test", f"{met['roc_auc']:.3f}".replace(".", ","),
                  help="Qué tan bien ordena el riesgo: 0,5 es azar y 1 es perfecto. La nota de Lending Club logra 0,683.")
        c2.metric("KS en test", f"{met['ks']:.3f}".replace(".", ","))
        c3.metric("PD media en test", pct(cal["pd_media"]), help="Después de calibrar.")
        c4.metric("Default real en test", pct(cal["default_observado"]))
    bandas = pd.DataFrame(modelo["bandas"])
    vista = pd.DataFrame({
        "Banda": bandas["banda"],
        "Rango de PD": [f"{pct(d)} a {pct(h)}" for d, h in zip(bandas["pd_desde"], bandas["pd_hasta"])],
        "Default en test": bandas["default_historico"].map(pct),
        "Severidad": bandas["severidad"].map(lambda s: pct(s, 1)),
        "Decisión": bandas["decision"].map(lambda d: DECISION[d][0])})
    st.dataframe(vista, use_container_width=True, hide_index=True)
