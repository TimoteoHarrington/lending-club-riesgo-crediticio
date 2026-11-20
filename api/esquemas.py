"""Entradas y salidas de la API (validadas con Pydantic)."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.solicitud import BURO, ESTADOS, PROPOSITOS, VIVIENDAS

MAX_LOTE = 1000

EJEMPLO = {
    "monto": 12000, "ingreso_anual": 65000, "dti": 18.5, "fico": 697,
    "antiguedad_crediticia_anios": 12, "antiguedad_laboral_anios": 4,
    "vivienda": "RENT", "proposito": "debt_consolidation", "estado": "CA",
    "acc_open_past_24mths": 5, "inq_last_6mths": 1,
}


class _SolicitudBase(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EJEMPLO})

    monto: float = Field(..., ge=500, le=40000, description="Monto pedido, en dólares (500 a 40.000)")
    ingreso_anual: float = Field(..., gt=0, le=10_000_000, description="Ingreso anual declarado, en dólares")
    dti: float = Field(..., ge=0, le=100, description="Deudas mensuales sobre ingreso mensual, en % (sin hipoteca)")
    fico: int = Field(..., ge=300, le=850, description="Puntaje FICO (300 a 850)")
    antiguedad_crediticia_anios: float = Field(..., ge=0, le=80, description="Años desde la primera línea de crédito")
    antiguedad_laboral_anios: Optional[int] = Field(
        None, ge=0, le=10, description="Años en el empleo actual (10 = 10 o más; vacío si no informa)")
    vivienda: Literal[tuple(VIVIENDAS)] = Field(..., description="RENT (alquila), MORTGAGE (hipoteca), OWN (propia), OTHER")
    proposito: Literal[tuple(PROPOSITOS)] = Field(..., description="Destino del préstamo")
    estado: Literal[tuple(ESTADOS)] = Field(..., description="Estado de residencia (dos letras)")
    mora_previa: Optional[bool] = Field(
        None, description="Si tuvo algún atraso alguna vez. Vacío: se deduce de los atrasos de los últimos 2 años")


# Los datos del buró son opcionales: si faltan, se completan con la mediana de train
Solicitud = create_model(
    "Solicitud",
    __base__=_SolicitudBase,
    **{col: (Optional[float], Field(None, ge=minimo, le=maximo, description=etiqueta))
       for col, (etiqueta, minimo, maximo) in BURO.items()},
)


class Lote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solicitudes: list[Solicitud] = Field(..., min_length=1, max_length=MAX_LOTE)


class Resultado(BaseModel):
    pd_modelo: float = Field(..., description="PD que da el modelo, sin calibrar")
    pd: float = Field(..., description="PD calibrada: probabilidad de default en los 36 meses")
    banda: str = Field(..., description="Banda de riesgo, de 1 (menor) a 7 (mayor)")
    default_historico_banda: Optional[float] = Field(None, description="Default observado en esa banda en test")
    severidad: float = Field(..., description="Pérdida si hay default, como % del monto (según la banda)")
    prima: float = Field(..., description="Pérdida esperada = PD × severidad, como % del monto, toda la vida del préstamo")
    perdida_esperada_usd: float = Field(..., description="Prima × monto, en dólares")
    decision: Literal["aprobar", "revisar", "rechazar"]
    motivo: str
    variables_completadas: list[str] = Field(..., description="Datos del buró que no vinieron y se completaron con la mediana de train")


class ResultadoLote(BaseModel):
    resultados: list[Resultado]
    resumen: dict[str, int] = Field(..., description="Cantidad de solicitudes por decisión")
