"""
De la PD del modelo a la prima de riesgo y la decisión (Entrega 04).

Pasos para cada préstamo:
1. Calibrar la PD con un ajuste de intercepto: se suma una constante a las
   log-odds. Sube o baja el nivel sin cambiar el orden (el AUC no cambia).
2. Asignar la banda de riesgo con los cortes de la Entrega 03, sobre la PD
   original. Como la calibración no cambia el orden, es lo mismo que usar la
   PD calibrada con los cortes calibrados.
3. Severidad de la banda: pérdida promedio de los defaults de esa banda, como
   % del monto prestado.
4. Prima = PD calibrada × severidad de la banda: la pérdida esperada de toda
   la vida del préstamo, como % del monto. No es una tasa anual.
5. Decisión por banda: aprobar, revisar o rechazar.

Ninguna función entrena ni lee datos: reciben los parámetros ya estimados
(notebook 04) y guardados en `modelos/modelo_pd.json`.
"""
import numpy as np
from scipy.optimize import brentq

EPSILON = 1e-6
DECISIONES = ("aprobar", "revisar", "rechazar")


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))


def sigmoide(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def calibrar_pd(pd_modelo, intercepto):
    """PD calibrada: misma forma, nivel corrido por `intercepto` (en log-odds)."""
    return sigmoide(logit(pd_modelo) + float(intercepto))


def estimar_intercepto(pd_modelo, y):
    """Constante que hace que la PD calibrada promedio iguale al default
    observado en `y`. Se estima con validación, nunca con test."""
    objetivo = float(np.mean(y))
    z = logit(pd_modelo)
    return float(brentq(lambda b: sigmoide(z + b).mean() - objetivo, -5, 5))


def asignar_banda(pd_modelo, cortes, nombres):
    """Banda de cada préstamo. Una PD igual a un corte cae en la banda de la derecha."""
    indice = np.searchsorted(np.asarray(cortes, dtype=float),
                             np.asarray(pd_modelo, dtype=float), side="right")
    return np.asarray(nombres)[indice]


def severidad_de_banda(bandas, severidad_por_banda):
    """Severidad (% del monto) de cada préstamo según su banda."""
    return np.array([severidad_por_banda[str(b)] for b in np.atleast_1d(bandas)], dtype=float)


def calcular_prima(pd_calibrada, severidad):
    """Pérdida esperada de toda la vida del préstamo, como % del monto prestado."""
    return np.asarray(pd_calibrada, dtype=float) * np.asarray(severidad, dtype=float)


def decidir(bandas, regla):
    """Decisión de cada préstamo según su banda.

    `regla` es un diccionario {"aprobar": [...], "revisar": [...], "rechazar": [...]}
    con los nombres de las bandas. Cada banda debe estar en una sola lista.
    """
    por_banda = {}
    for decision in DECISIONES:
        for banda in regla.get(decision, []):
            if str(banda) in por_banda:
                raise ValueError(f"La banda {banda} aparece en más de una decisión.")
            por_banda[str(banda)] = decision
    try:
        return np.array([por_banda[str(b)] for b in np.atleast_1d(bandas)])
    except KeyError as error:
        raise ValueError(f"La regla de decisión no incluye la banda {error.args[0]}.") from None
