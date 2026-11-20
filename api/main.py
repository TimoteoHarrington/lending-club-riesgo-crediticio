"""
API REST del modelo de riesgo de default (Entrega 04).

La API solo recibe, valida y responde: la lógica está en `src/` (evaluador,
solicitud y pricing). El modelo se carga una vez, al arrancar.

Correr en local:
    uvicorn api.main:app --reload
Documentación interactiva: http://localhost:8000/docs
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.esquemas import MAX_LOTE, Lote, Resultado, ResultadoLote, Solicitud  # noqa: E402
from src.evaluador import EvaluadorRiesgo  # noqa: E402
from src.solicitud import BURO, ESTADOS, PROPOSITOS, VIVIENDAS  # noqa: E402

log = logging.getLogger("api")
VERSION = "1.0.0"


@asynccontextmanager
async def ciclo_de_vida(app):
    try:
        app.state.evaluador = EvaluadorRiesgo()
        log.info("Modelo cargado")
    except Exception:  # la API arranca igual y /salud informa el problema
        log.exception("No se pudo cargar el modelo")
        app.state.evaluador = None
    yield


app = FastAPI(
    title="API de riesgo de default · Lending Club",
    description=(
        "Recibe una solicitud de préstamo y devuelve la probabilidad de default (PD) calibrada, "
        "la banda de riesgo, la prima (PD × severidad) y la decisión. "
        "Proyecto de Ciencia de Datos Aplicada (ITBA)."),
    version=VERSION,
    lifespan=ciclo_de_vida,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def evaluador(request: Request) -> EvaluadorRiesgo:
    ev = request.app.state.evaluador
    if ev is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible. Revisar los logs del servicio.")
    return ev


# ---------------------------------------------------------------------------
# Errores: siempre JSON con un mensaje en castellano
# ---------------------------------------------------------------------------

MENSAJES = {
    "missing": "Falta este dato.",
    "extra_forbidden": "Campo desconocido.",
    "float_parsing": "Tiene que ser un número.",
    "float_type": "Tiene que ser un número.",
    "int_parsing": "Tiene que ser un número entero.",
    "int_type": "Tiene que ser un número entero.",
    "int_from_float": "Tiene que ser un número entero.",
    "bool_parsing": "Tiene que ser verdadero o falso.",
    "too_short": "La lista no puede estar vacía.",
    "too_long": f"Se pueden evaluar hasta {MAX_LOTE} solicitudes por pedido.",
    "json_invalid": "El cuerpo del pedido no es un JSON válido.",
}


def _mensaje(error):
    tipo, ctx = error["type"], error.get("ctx", {})
    if tipo == "greater_than_equal":
        return f"Tiene que ser mayor o igual a {ctx['ge']}."
    if tipo == "greater_than":
        return f"Tiene que ser mayor que {ctx['gt']}."
    if tipo == "less_than_equal":
        return f"Tiene que ser menor o igual a {ctx['le']}."
    if tipo == "literal_error":
        return f"Valor no permitido. Opciones: {ctx.get('expected', '')}."
    return MENSAJES.get(tipo, error.get("msg", "Valor inválido."))


@app.exception_handler(RequestValidationError)
async def error_de_validacion(request, exc):
    detalles = [{"campo": ".".join(str(p) for p in e["loc"] if p != "body") or "cuerpo",
                 "mensaje": _mensaje(e)} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Datos inválidos", "detalles": detalles})


@app.exception_handler(HTTPException)
async def error_http(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(ValueError)
async def error_de_datos(request, exc):
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(Exception)
async def error_inesperado(request, exc):
    log.exception("Error inesperado")
    return JSONResponse(status_code=500, content={"error": "Error interno al evaluar la solicitud."})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
def inicio():
    return {"servicio": app.title, "version": VERSION, "documentacion": "/docs",
            "endpoints": ["/salud", "/modelo", "/opciones", "/evaluar", "/evaluar/lote"]}


@app.get("/salud", tags=["info"])
def salud(request: Request):
    """Estado del servicio. Sirve para comprobar que la API está levantada."""
    ev = request.app.state.evaluador
    return {"estado": "ok" if ev else "sin modelo", "modelo_cargado": ev is not None,
            "parametros_completos": bool(ev and ev.parametros_completos), "version": VERSION}


@app.get("/modelo", tags=["info"])
def modelo(request: Request):
    """Modelo, métricas de test, bandas con su severidad y decisión, y calibración."""
    return evaluador(request).info()


@app.get("/opciones", tags=["info"])
def opciones():
    """Valores válidos para armar un formulario de solicitud."""
    return {"viviendas": VIVIENDAS, "propositos": PROPOSITOS, "estados": ESTADOS,
            "buro": [{"campo": c, "etiqueta": e, "minimo": mi, "maximo": ma}
                     for c, (e, mi, ma) in BURO.items()]}


@app.post("/evaluar", response_model=Resultado, tags=["evaluación"])
def evaluar(solicitud: Solicitud, request: Request):
    """Evalúa una solicitud: PD calibrada, banda, prima y decisión."""
    return evaluador(request).evaluar([solicitud.model_dump()])[0]


@app.post("/evaluar/lote", response_model=ResultadoLote, tags=["evaluación"])
def evaluar_lote(lote: Lote, request: Request):
    """Evalúa varias solicitudes a la vez (hasta 1000)."""
    resultados = evaluador(request).evaluar([s.model_dump() for s in lote.solicitudes])
    resumen = {d: sum(r["decision"] == d for r in resultados) for d in ("aprobar", "revisar", "rechazar")}
    return {"resultados": resultados, "resumen": resumen}
