"""Tests de la lógica de E04: calibración, prima, decisión y armado de la solicitud.

No usan los datos de Kaggle: casos sintéticos y el modelo guardado.
Ejecutar: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np
import pandas as pd

from src import features as F
from src import pricing as P
from src.evaluacion import asignar_banda as asignar_banda_e03
from src.evaluador import EvaluadorRiesgo
from src.solicitud import BURO, construir_variables

SOLICITUD = {
    "monto": 12000, "ingreso_anual": 65000, "dti": 18.5, "fico": 697,
    "antiguedad_crediticia_anios": 12.0, "antiguedad_laboral_anios": 4,
    "vivienda": "RENT", "proposito": "debt_consolidation", "estado": "CA",
}


class PricingTests(unittest.TestCase):
    def test_calibrar_con_intercepto_cero_no_cambia_nada(self):
        p = np.array([0.01, 0.1, 0.5, 0.9])
        np.testing.assert_allclose(P.calibrar_pd(p, 0.0), p, rtol=1e-9)

    def test_calibrar_conserva_el_orden_y_sube_el_nivel(self):
        p = np.linspace(0.01, 0.6, 50)
        calibrada = P.calibrar_pd(p, 0.1)
        self.assertTrue((np.diff(calibrada) > 0).all())
        self.assertTrue((calibrada > p).all())
        self.assertTrue(((calibrada > 0) & (calibrada < 1)).all())

    def test_intercepto_iguala_el_promedio_al_default(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.4, 5000)
        y = rng.random(5000) < p * 1.08
        b = P.estimar_intercepto(p, y)
        self.assertAlmostEqual(P.calibrar_pd(p, b).mean(), y.mean(), places=6)
        self.assertGreater(b, 0)

    def test_bandas_iguales_a_las_de_e03(self):
        cortes, nombres = [0.1, 0.2, 0.3], ["1", "2", "3", "4"]
        p = np.array([0.05, 0.1, 0.15, 0.2, 0.31, 0.99])
        np.testing.assert_array_equal(
            P.asignar_banda(p, cortes, nombres), np.asarray(asignar_banda_e03(p, cortes, nombres)))

    def test_prima_es_pd_por_severidad(self):
        sev = P.severidad_de_banda(["1", "2"], {"1": 0.5, "2": 0.6})
        np.testing.assert_allclose(P.calcular_prima([0.1, 0.2], sev), [0.05, 0.12])

    def test_decision_y_errores_de_la_regla(self):
        regla = {"aprobar": ["1"], "revisar": ["2"], "rechazar": ["3"]}
        self.assertEqual(list(P.decidir(["1", "3"], regla)), ["aprobar", "rechazar"])
        with self.assertRaises(ValueError):
            P.decidir(["4"], regla)
        with self.assertRaises(ValueError):
            P.decidir(["1"], {"aprobar": ["1"], "rechazar": ["1"]})


class SolicitudTests(unittest.TestCase):
    """La solicitud debe dar las mismas variables que construir_features (E02/E03)."""

    def test_igual_a_construir_features(self):
        numericas, binarias, categoricas = F.features_modelo()
        rng = np.random.default_rng(1)
        crudo = {col: float(rng.integers(1, 30)) for col in F.BURO_BASICO + F.BURO_EXTENDIDO}
        crudo.update(
            loan_amnt=12000.0, annual_inc=65000.0, dti=18.5, fico_range_low=695.0,
            fico_range_high=699.0, issue_d="Oct-2015", earliest_cr_line="Oct-2003",
            emp_length="4 years", home_ownership="RENT", purpose="debt_consolidation",
            addr_state="CA", mths_since_last_delinq=np.nan, mths_since_last_record=np.nan,
            delinq_2yrs=0.0, pub_rec=1.0)
        esperado = F.construir_features(pd.DataFrame([crudo])).iloc[0]

        solicitud = dict(SOLICITUD, antiguedad_crediticia_anios=esperado["antiguedad_crediticia_anios"])
        solicitud.update({col: crudo[col] for col in BURO})
        defectos = {col: esperado[col] for col in numericas}
        obtenido, completadas = construir_variables(solicitud, numericas, binarias, categoricas, defectos)

        self.assertEqual(completadas, [])
        self.assertEqual(list(obtenido), numericas + binarias + categoricas)
        for col in obtenido:
            if col in categoricas:
                self.assertEqual(obtenido[col], esperado[col], col)
            else:
                self.assertAlmostEqual(float(obtenido[col]), float(esperado[col]), places=9, msg=col)

    def test_completa_el_buro_y_marca_lo_que_falta(self):
        numericas, binarias, categoricas = F.features_modelo()
        solicitud = dict(SOLICITUD, antiguedad_laboral_anios=None)
        variables, completadas = construir_variables(
            solicitud, numericas, binarias, categoricas, {"revol_bal": 9000.0})
        self.assertEqual(sorted(completadas), sorted(BURO))
        self.assertEqual(variables["revol_bal"], 9000.0)
        self.assertEqual(variables["emp_length_faltante"], 1)
        self.assertTrue(np.isnan(variables["emp_length_anios"]))

    def test_errores_claros(self):
        numericas, binarias, categoricas = F.features_modelo()
        with self.assertRaisesRegex(ValueError, "fico"):
            construir_variables({k: v for k, v in SOLICITUD.items() if k != "fico"},
                                numericas, binarias, categoricas, {})
        with self.assertRaisesRegex(ValueError, "ingreso"):
            construir_variables(dict(SOLICITUD, ingreso_anual=0), numericas, binarias, categoricas, {})


class EvaluadorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluador = EvaluadorRiesgo()

    def test_resultado_completo_y_coherente(self):
        r = self.evaluador.evaluar([SOLICITUD])[0]
        for clave in ("pd_modelo", "pd", "banda", "severidad", "prima", "decision", "motivo",
                      "perdida_esperada_usd", "variables_completadas"):
            self.assertIn(clave, r)
        self.assertTrue(0 < r["pd"] < 1)
        self.assertAlmostEqual(r["prima"], r["pd"] * r["severidad"], places=5)
        self.assertIn(r["decision"], P.DECISIONES)

    def test_perfil_riesgoso_tiene_mas_pd_que_perfil_seguro(self):
        seguro = dict(SOLICITUD, fico=800, dti=5, monto=5000, ingreso_anual=150000,
                      vivienda="MORTGAGE", acc_open_past_24mths=1, inq_last_6mths=0)
        riesgoso = dict(SOLICITUD, fico=662, dti=35, monto=30000, ingreso_anual=40000,
                        acc_open_past_24mths=14, inq_last_6mths=5, proposito="small_business")
        r_seguro, r_riesgoso = self.evaluador.evaluar([seguro, riesgoso])
        self.assertLess(r_seguro["pd"], r_riesgoso["pd"])
        self.assertLessEqual(int(r_seguro["banda"]), int(r_riesgoso["banda"]))

    def test_info_tiene_todas_las_bandas(self):
        info = self.evaluador.info()
        self.assertEqual([b["banda"] for b in info["bandas"]], self.evaluador.bandas)


if __name__ == "__main__":
    unittest.main()
