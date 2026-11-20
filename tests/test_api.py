"""Tests de la API: respuestas correctas y errores claros.

Ejecutar: python -m unittest discover -s tests -v
"""
import unittest

from fastapi.testclient import TestClient

from api.esquemas import EJEMPLO
from api.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cliente = TestClient(app)
        cls.cliente.__enter__()  # ejecuta el arranque: carga el modelo

    @classmethod
    def tearDownClass(cls):
        cls.cliente.__exit__(None, None, None)

    def test_salud(self):
        r = self.cliente.get("/salud")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["modelo_cargado"])

    def test_modelo_y_opciones(self):
        self.assertEqual(len(self.cliente.get("/modelo").json()["bandas"]), 7)
        opciones = self.cliente.get("/opciones").json()
        self.assertIn("CA", opciones["estados"])
        self.assertTrue(opciones["buro"])

    def test_evaluar(self):
        r = self.cliente.post("/evaluar", json=EJEMPLO)
        self.assertEqual(r.status_code, 200, r.text)
        datos = r.json()
        self.assertIn(datos["decision"], ("aprobar", "revisar", "rechazar"))
        self.assertAlmostEqual(datos["prima"], datos["pd"] * datos["severidad"], places=5)
        self.assertNotIn("acc_open_past_24mths", datos["variables_completadas"])
        self.assertIn("revol_bal", datos["variables_completadas"])

    def test_lote(self):
        r = self.cliente.post("/evaluar/lote", json={"solicitudes": [EJEMPLO, dict(EJEMPLO, fico=780)]})
        self.assertEqual(r.status_code, 200, r.text)
        datos = r.json()
        self.assertEqual(len(datos["resultados"]), 2)
        self.assertEqual(sum(datos["resumen"].values()), 2)

    def test_errores_de_validacion_en_castellano(self):
        malo = dict(EJEMPLO, fico=900, estado="XX")
        del malo["monto"]
        r = self.cliente.post("/evaluar", json=malo)
        self.assertEqual(r.status_code, 422)
        campos = {d["campo"]: d["mensaje"] for d in r.json()["detalles"]}
        self.assertEqual(campos["monto"], "Falta este dato.")
        self.assertIn("menor o igual a 850", campos["fico"])
        self.assertIn("Valor no permitido", campos["estado"])

    def test_campo_desconocido_y_lote_vacio(self):
        r = self.cliente.post("/evaluar", json=dict(EJEMPLO, sueldo=1))
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detalles"][0]["mensaje"], "Campo desconocido.")
        self.assertEqual(self.cliente.post("/evaluar/lote", json={"solicitudes": []}).status_code, 422)

    def test_json_invalido(self):
        r = self.cliente.post("/evaluar", content="no es json", headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("error", r.json())


if __name__ == "__main__":
    unittest.main()
