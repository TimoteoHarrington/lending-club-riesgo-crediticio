"""
De una solicitud de préstamo a las 54 variables que espera el modelo.

Una persona que pide un préstamo no conoce las 54 variables del modelo. La
solicitud tiene dos partes:

- Datos principales (obligatorios): monto, ingreso, DTI, FICO, antigüedad
  crediticia y laboral, vivienda, destino del préstamo y estado.
- Datos del buró (opcionales): las variables de historial crediticio que más
  usa el modelo. Si no se informan, se completan con la mediana de train.

El resto de las variables del modelo también se completa con la mediana de
train (calculada en el notebook 04). Las transformaciones replican las de
`src.features.construir_features`, que es la referencia: hay un test que
compara las dos.
"""
import numpy as np

# Datos principales: nombre en la solicitud -> descripción
PRINCIPALES = {
    "monto": "Monto pedido, en dólares",
    "ingreso_anual": "Ingreso anual declarado, en dólares",
    "dti": "Deudas mensuales sobre ingreso mensual, en % (sin hipoteca)",
    "fico": "Puntaje FICO",
    "antiguedad_crediticia_anios": "Años desde la primera línea de crédito",
    "antiguedad_laboral_anios": "Años en el empleo actual (0 a 10; vacío si no informa)",
    "vivienda": "RENT, MORTGAGE, OWN u OTHER",
    "proposito": "Destino del préstamo (códigos de Lending Club)",
    "estado": "Estado de residencia (código de dos letras)",
}

# Datos del buró que se pueden informar: variable del modelo -> (etiqueta, mínimo, máximo)
BURO = {
    "acc_open_past_24mths": ("Cuentas abiertas en los últimos 24 meses", 0, 100),
    "num_tl_op_past_12m": ("Cuentas abiertas en los últimos 12 meses", 0, 100),
    "inq_last_6mths": ("Consultas al buró en los últimos 6 meses", 0, 50),
    "revol_bal": ("Saldo en tarjetas y crédito rotativo (USD)", 0, 5_000_000),
    "revol_util": ("Uso del crédito rotativo (%)", 0, 200),
    "bc_open_to_buy": ("Disponible en tarjetas (USD)", 0, 5_000_000),
    "total_bc_limit": ("Límite total en tarjetas (USD)", 0, 5_000_000),
    "tot_hi_cred_lim": ("Límite total de crédito (USD)", 0, 20_000_000),
    "open_acc": ("Cuentas de crédito abiertas", 0, 100),
    "total_acc": ("Cuentas de crédito totales en su historia", 0, 200),
    "mort_acc": ("Hipotecas", 0, 50),
    "delinq_2yrs": ("Atrasos de 30+ días en los últimos 2 años", 0, 50),
    "pub_rec": ("Registros públicos negativos", 0, 50),
    "pub_rec_bankruptcies": ("Quiebras", 0, 20),
}

VIVIENDAS = ["RENT", "MORTGAGE", "OWN", "OTHER"]
PROPOSITOS = ["debt_consolidation", "credit_card", "home_improvement", "other",
              "major_purchase", "medical", "small_business", "car", "moving",
              "vacation", "house", "wedding", "renewable_energy", "educational"]
ESTADOS = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA",
           "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS",
           "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA",
           "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]


def construir_variables(solicitud, numericas, binarias, categoricas, valores_por_defecto):
    """Variables del modelo para una solicitud.

    Devuelve (variables, completadas): un diccionario con las 54 variables y la
    lista de variables del buró que no vinieron en la solicitud y se completaron
    con la mediana de train. Si una mediana no está disponible queda vacía (NaN),
    que LightGBM trata como dato faltante.
    """
    faltan = [c for c in PRINCIPALES if c != "antiguedad_laboral_anios" and solicitud.get(c) is None]
    if faltan:
        raise ValueError(f"Faltan datos principales: {', '.join(faltan)}")
    ingreso = float(solicitud["ingreso_anual"])
    if ingreso <= 0:
        raise ValueError("El ingreso anual debe ser mayor que 0.")

    variables = {col: valores_por_defecto.get(col, np.nan) for col in numericas}
    completadas = []
    for col in BURO:
        valor = solicitud.get(col)
        if valor is None:
            completadas.append(col)
        else:
            variables[col] = float(valor)

    monto = float(solicitud["monto"])
    laboral = solicitud.get("antiguedad_laboral_anios")
    variables.update(
        loan_amnt=monto,
        dti=float(solicitud["dti"]) if float(solicitud["dti"]) <= 100 else np.nan,
        fico_promedio=float(solicitud["fico"]),
        log_annual_inc=float(np.log(ingreso)),
        antiguedad_crediticia_anios=float(solicitud["antiguedad_crediticia_anios"]),
        emp_length_anios=np.nan if laboral is None else float(min(max(int(laboral), 0), 10)),
        prestamo_sobre_ingreso=monto / ingreso,
        revol_bal_sobre_ingreso=float(variables["revol_bal"]) / ingreso,
    )

    mora_previa = solicitud.get("mora_previa")
    if mora_previa is None:
        mora_previa = (variables.get("delinq_2yrs") or 0) > 0
    variables.update(
        tuvo_mora_previa=int(bool(mora_previa)),
        tuvo_registro_publico=int((variables.get("pub_rec") or 0) > 0),
        emp_length_faltante=int(laboral is None),
        buro_extendido_disponible=1,
    )

    vivienda = str(solicitud["vivienda"]).upper()
    variables.update(
        home_ownership=vivienda if vivienda in VIVIENDAS[:3] else "OTHER",
        purpose=str(solicitud["proposito"]),
        addr_state=str(solicitud["estado"]).upper(),
    )
    orden = list(numericas) + list(binarias) + list(categoricas)
    return {col: variables[col] for col in orden}, completadas
