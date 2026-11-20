"""
Evaluador de solicitudes: une el modelo guardado con la calibración, las
bandas, la severidad, la prima y la regla de decisión.

Es la lógica del sistema, independiente de cómo se la llame: la usan la API
(`api/`), los tests y el notebook 04. No entrena nada.
"""
import numpy as np
import pandas as pd

from src import pricing as P
from src.modelado import cargar_modelo, predecir_pd
from src.solicitud import construir_variables

REGLA_POR_DEFECTO = {"aprobar": ["1", "2", "3", "4"], "revisar": ["5"], "rechazar": ["6", "7"]}

MOTIVOS = {
    "aprobar": "Riesgo dentro de la política: se aprueba con la prima indicada.",
    "revisar": "Riesgo alto: lo tiene que revisar un analista antes de aprobar.",
    "rechazar": "Riesgo fuera de la política: la pérdida esperada es demasiado alta.",
}


class EvaluadorRiesgo:
    """Carga el modelo y sus parámetros una sola vez y evalúa solicitudes."""

    def __init__(self, nombre="modelo_pd"):
        self.modelo, self.meta = cargar_modelo(nombre)
        variables = self.meta["variables"]
        self.numericas = variables["numericas"]
        self.binarias = variables["binarias"]
        self.categoricas = variables["categoricas"]
        self.bandas = self.meta["bandas"]["nombres"]
        self.cortes = self.meta["bandas"]["cortes_pd"]

        calibracion = self.meta.get("calibracion", {})
        self.intercepto = float(calibracion.get("intercepto", 0.0))
        severidad_unica = float(self.meta["severidad_sobre_monto"])
        self.severidad_por_banda = {
            b: float(self.meta.get("severidad_por_banda", {}).get(b, severidad_unica))
            for b in self.bandas}
        self.regla = self.meta.get("regla_decision", REGLA_POR_DEFECTO)
        self.valores_por_defecto = self.meta.get("valores_por_defecto", {})
        self.default_por_banda = self.meta.get("default_por_banda", {})
        # Verifica la regla al arrancar, no en la primera solicitud
        P.decidir(self.bandas, self.regla)

    @property
    def parametros_completos(self):
        """True si el notebook 04 ya guardó calibración, severidades y medianas."""
        return all(k in self.meta for k in
                   ("calibracion", "severidad_por_banda", "valores_por_defecto"))

    def variables(self, solicitudes):
        """DataFrame con las 54 variables del modelo y la lista de completadas."""
        filas, completadas = [], []
        for solicitud in solicitudes:
            fila, faltantes = construir_variables(
                solicitud, self.numericas, self.binarias, self.categoricas,
                self.valores_por_defecto)
            filas.append(fila)
            completadas.append(faltantes)
        return pd.DataFrame(filas), completadas

    def evaluar(self, solicitudes):
        """Evalúa una lista de solicitudes (diccionarios) y devuelve una lista de resultados."""
        if not solicitudes:
            return []
        entrada, completadas = self.variables(solicitudes)
        pd_modelo = predecir_pd(self.modelo, entrada)
        pd_calibrada = P.calibrar_pd(pd_modelo, self.intercepto)
        bandas = P.asignar_banda(pd_modelo, self.cortes, self.bandas)
        severidad = P.severidad_de_banda(bandas, self.severidad_por_banda)
        prima = P.calcular_prima(pd_calibrada, severidad)
        decisiones = P.decidir(bandas, self.regla)

        resultados = []
        for i, solicitud in enumerate(solicitudes):
            monto = float(solicitud["monto"])
            banda = str(bandas[i])
            resultados.append({
                "pd_modelo": round(float(pd_modelo[i]), 6),
                "pd": round(float(pd_calibrada[i]), 6),
                "banda": banda,
                "default_historico_banda": self.default_por_banda.get(banda),
                "severidad": round(float(severidad[i]), 6),
                "prima": round(float(prima[i]), 6),
                "perdida_esperada_usd": round(float(prima[i]) * monto, 2),
                "decision": str(decisiones[i]),
                "motivo": MOTIVOS[str(decisiones[i])],
                "variables_completadas": completadas[i],
            })
        return resultados

    def info(self):
        """Resumen del modelo y de los parámetros de la prima, para la API y la interfaz."""
        # Límites de cada banda en la escala de la PD calibrada (la que ve el usuario)
        cortes = [float(c) for c in P.calibrar_pd(self.cortes, self.intercepto)]
        return {
            "modelo": self.meta["modelo"],
            "descripcion": self.meta["descripcion"],
            "entrenado_con": self.meta["entrenado_con"],
            "bandas": [{
                "banda": b,
                "pd_desde": 0.0 if i == 0 else cortes[i - 1],
                "pd_hasta": 1.0 if i == len(self.bandas) - 1 else cortes[i],
                "severidad": self.severidad_por_banda[b],
                "default_historico": self.default_por_banda.get(b),
                "decision": str(P.decidir([b], self.regla)[0]),
            } for i, b in enumerate(self.bandas)],
            "calibracion": self.meta.get("calibracion", {"metodo": "sin calibrar", "intercepto": 0.0}),
            "metricas_test": self.meta.get("metricas_test", {}),
            "metricas_test_calibrado": self.meta.get("metricas_test_calibrado", {}),
            "parametros_completos": self.parametros_completos,
        }
