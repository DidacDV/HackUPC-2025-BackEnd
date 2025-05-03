import polars as pl
import os
import numpy as np
from pathlib import Path
from unidecode import unidecode
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

def get_Path(filename, tipo) -> str:
    if not filename.endswith(tipo):
        filename = f"{filename}.{tipo}"
    
    current_dir = Path(__file__).parent
    csv_dir = current_dir / 'DataSets'
    file_path = csv_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filename}")
    
    return file_path

#Transformo data float  
def ProcesamientoInicial(data):
    data.columns = [
            unidecode(col.strip().lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("º", "n")) for col in data.columns]
    
    for col in data.columns:
        if data[col].dtype == pl.Utf8:
            data = data.with_columns(
                pl.col(col)
                .map_elements(lambda x: unidecode(x.strip().lower()) if x is not None else x, return_dtype=pl.Utf8)
                .alias(col))

    df = data.with_columns([
        pl.col("importe_orig.").cast(pl.Float64, strict=False),
        pl.col("importe_pagado").cast(pl.Float64, strict=False),
        pl.col("comision").cast(pl.Float64, strict=False),
        pl.col("tasa_impuestos").cast(pl.Float64, strict=False),
        pl.col("imp._impuestos").cast(pl.Float64, strict=False)
    ])
    return df

def limpiarDataSet(data):
    columnas = data.columns

    #Eliminamos null
    df = data.drop_nulls(columnas)

    #Horas correctas
    df = df.filter((pl.col("fin_transaccion_utc") >= pl.col("hora_inicio_utc")))
    df = df.filter(pl.col("fecha_envio_utc") >= pl.col("fin_transaccion_utc"))

    #Id transaccion unica
    df = df.unique(subset=["id_transaccion"])

    #Importes Negativos
    #Ingreso y gasto simultaneo
    df = df.filter(
        ((pl.col("ingreso") > 0) & (pl.col("gasto") == 0.0)) |
        ((pl.col("gasto") > 0) & (pl.col("ingreso") == 0.0)))


    return df

def Categorias(data):
    #Caracterizacion
    dinero = data.select([pl.col("id_transaccion"), pl.col("ingreso"), pl.col("gasto"), pl.col("id_gasto")])
    tiempoTransaccion = data.select(([pl.col("id_transaccion"), (pl.col("fin_transaccion_utc") - pl.col("hora_inicio_utc")).alias("tiempo_transaccion")]))
    tipoTransaccion = data.select([pl.col("id_transaccion"), pl.col("tipo"), pl.col("descripcion"), pl.col("descripcion_gasto")])
    categoria = data.select([pl.col("id_transaccion"), pl.col("categoria"), pl.col("codigo_categoria")])
    mismaDivisa = data.select([pl.col("id_transaccion"), (pl.col("divisa_original") == pl.col("moneda_de_pago")).alias("misma_divisa")])
    infoPagador = data.select([pl.col("id_transaccion"), pl.col("nn_tarjeta"), pl.col("pagador")])
    frecuenciaTemporal = (data.with_columns([pl.col("hora_inicio_utc").dt.date().alias("dia")]).group_by("dia").agg([pl.len().alias("n_transacciones")]).sort("dia"),
                          data.with_columns([pl.col("hora_inicio_utc").dt.strftime("%Y-%m").alias("mes")]).group_by("mes").agg([pl.len().alias("n_transacciones")]).sort("mes"),
                          data.with_columns([pl.col("hora_inicio_utc").dt.strftime("%Y").alias("año")]).group_by("año").agg([pl.len().alias("n_transacciones")]).sort("año")
                        )
    motivo = (data.select([pl.col("id_transaccion"), pl.col("explicacion_anomalia")]))
    motivo_cat = (data.select([pl.col("id_transaccion"), pl.col("mensaje_personalizado")]))
    return (dinero, tipoTransaccion, categoria, mismaDivisa, infoPagador, frecuenciaTemporal, motivo, motivo_cat)

def guardarEnCSV(path, tuple, nombreArchivo):
    dinero, tipoTransaccion, categoria, mismaDivisa, infoPagador, frecuenciaTemporal, motivo, motivo_cat = tuple

    os.makedirs(path, exist_ok=True)

    dinero.write_csv(f"{path}/{nombreArchivo}Dinero.csv")
    tipoTransaccion.write_csv(f"{path}/{nombreArchivo}tipo_transaccion.csv")
    categoria.write_csv(f"{path}/{nombreArchivo}categoria.csv")
    mismaDivisa.write_csv(f"{path}/{nombreArchivo}misma_divisa.csv")
    infoPagador.write_csv(f"{path}/{nombreArchivo}info_pagador.csv")

    frecuenciaTemporal[0].write_csv(f"{path}/{nombreArchivo}frecuencia_dia.csv")
    frecuenciaTemporal[1].write_csv(f"{path}/{nombreArchivo}frecuencia_mes.csv")
    frecuenciaTemporal[2].write_csv(f"{path}/{nombreArchivo}frecuencia_year.csv")
    motivo.write_csv(f"{path}/{nombreArchivo}motivo.csv")
    motivo_cat.write_csv(f"{path}/{nombreArchivo}motivo2.csv")


def exportarArchivos(tuple):
    current_dir = Path(__file__).parent
    csv_dir = current_dir / 'DataSets'
    guardarEnCSV(csv_dir, tuple, "E1")

def crearNuevosDataSets(base):
    tipoCorrecto = ProcesamientoInicial(base)
    setLimpaido = limpiarDataSet(tipoCorrecto)
    categorias = set(setLimpaido["categoria"].unique().to_list())    
    
    #esto tiene que devolver NuevaData despues de los analisis 
    return setLimpaido, categorias

def IsolationAnalisis(data):
    columnas = ["hora_inicio_utc", "fin_transaccion_utc", "importe_orig.", "importe_pagado", "comision", "tasa_impuestos", "imp._impuestos", "ingreso", "gasto"]

    X = data.select(columnas).to_numpy()

    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    isolation_forest = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # Esperamos aproximadamente un 10% de anomalías
        random_state=42)
    isolation_forest.fit(X_scaled)

    # Predecir anomalías (-1 para anomalías, 1 para observaciones normales)
    predictions = isolation_forest.predict(X_scaled)
    anomaly_scores = isolation_forest.decision_function(X_scaled)

    df = data.with_columns([
        pl.Series(name="anomalia_numerica", values=(predictions == -1).astype(int)),
        pl.Series(name="score_anomalia", values=anomaly_scores)
    ])

    df = df.with_columns(
        pl.struct(["anomalia_numerica", "score_anomalia"] + columnas)
        .map_elements(lambda x: generate_anomaly_explanation(x))
        .alias("explicacion_anomalia")
    )

    return df

def generate_anomaly_explanation(row):
    if row["anomalia_numerica"] == 0:
        return "Normal transaction"
    
    # Cargar modelo de lenguaje (puedes usar uno más pequeño si es necesario)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Crear un texto descriptivo de la transacción
    transaction_details = (
        f"Transaction with original amount {row['importe_orig.']} and amount paid {row['importe_pagado']}. "
        f"Commission: {row['comision']}, Tax rate: {row['tasa_impuestos']}, "
        f"Income: {row['ingreso']}, Spent: {row['gasto']}. "
        f"Anomaly score: {row['score_anomalia']:.2f}"
    )
    
    # Generar un embedding del texto
    embedding = model.encode(transaction_details)
    
    # Definir posibles razones de anomalías (podrías expandir esto)
    common_reasons = [
        "The amount paid is significantly different from the original amount",
        "The commission is unusually high for this type of transaction.",
        "The tax rate does not match the expected values",
        "The relationship between income and expenditure is atypical",
        "The transaction duration is anomalous",
        "Unusual transaction pattern compared to historical behavior",
        "The discount applied is unexpectedly high or low",
        "The commission varies drastically without a logical explanation",
        "The amount of taxes is inconsistent with the rate applied",
        "Unusual rounding is observed in the amounts or commissions",
        "An income or expense is recorded outside the expected range for the time or date",
        "The proportion of expenditure in a specific transaction is excessively high or low compared to the income generated",
        "Income or expenses with identical values ​​are recorded repeatedly",
        "Transactions with similar amounts have extremely different durations",
        "The transaction duration is unusually short or long for the type of product or service",
        "A sudden and significant increase or decrease in the average transaction value",
        "An unusual frequency of transactions with specific values ​​(unexpected peaks or valleys)",
        "Drastic changes in the distribution of values ​​in numeric columns"
    ]
    
    # Encontrar la razón más similar
    reason_embeddings = model.encode(common_reasons)
    similarities = cosine_similarity([embedding], reason_embeddings)
    best_match_idx = np.argmax(similarities)
    best_match = common_reasons[best_match_idx]
    
    # Construir el mensaje final
    explanation = (
        f"POSSIBLE ANOMALY DETECTED (score:{row['score_anomalia']:.2f}). "
        f"Most likely reason: {best_match}. "
        f"Details: {transaction_details}"
    )
    
    return explanation


def evaluarSemantica(data: pl.DataFrame, categorias: list[str]) -> pl.DataFrame:
    try:
        embed_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Modelo ligero (33MB)
        print("Modelo de embeddings cargado")
    except Exception as e:
        raise RuntimeError(f"Error cargando embeddings: {e}")

    category_embeddings = {cat: embed_model.encode(cat) for cat in categorias}
    
    keywords_por_categoria = {
        "Alimentación": ["supermercado", "comida", "restaurante", "alimento"],
        "Transporte": ["gasolina", "taxi", "autobús", "metro", "transporte"],
        "Entretenimiento": ["cine", "netflix", "streaming", "videojuego", "ocio"],
        "Servicios": ["luz", "agua", "internet", "teléfono", "factura"]
    }

    results = []
    for row in data.rows(named=True):
        cat_actual, desc = row["categoria"], row["descripcion_gasto"]
        
        if not cat_actual or not desc:
            results.append((0.0, False, "Incomplete data", ""))
            continue
        
        try:
            desc_embed = embed_model.encode(desc)
            sim = cosine_similarity([desc_embed], [category_embeddings[cat_actual]])[0][0]
            is_anomaly = sim < 0.05  
            
            msg = ""
            if is_anomaly:
                similitudes = {
                    cat: cosine_similarity([desc_embed], [emb])[0][0]
                    for cat, emb in category_embeddings.items()
                }
                mejor_cat = max(similitudes.items(), key=lambda x: x[1])[0]
                
                terminos_clave = []
                for cat, keywords in keywords_por_categoria.items():
                    if any(keyword in desc.lower() for keyword in keywords):
                        terminos_clave.extend(keywords)
                
                if terminos_clave:
                    terminos_str = ", ".join(f"'{t}'" for t in set(terminos_clave))
                    msg = (f"Possible error: The description contains terms ({terminos_str}) "
                          f"that suggest the category '{mejor_cat}' rather '{cat_actual}'")
                else:
                    msg = (f"Possible error: The description does not match '{cat_actual}'. "
                          f"Suggestion: consider '{mejor_cat}' (similarity: {similitudes[mejor_cat]:.2f})")
            
            results.append((float(sim), is_anomaly, f"Similarity: {sim:.2f}", msg))
        except Exception as e:
            results.append((0.0, False, f"Error: {str(e)}", ""))
    
    return data.with_columns([
        pl.Series(name="coherencia_semantica", values=[r[0] for r in results]),
        pl.Series(name="anomalia_semantica", values=[r[1] for r in results]),
        pl.Series(name="explicacion_semantica", values=[r[2] for r in results]),
        pl.Series(name="mensaje_personalizado", values=[r[3] for r in results])
    ])

if __name__ == "__main__":
    
    try:
        path = get_Path("E1", "csv")
        data = pl.read_csv(path, try_parse_dates=True)

        final, categorias = crearNuevosDataSets(data)
        info = IsolationAnalisis(final)
        acabado = evaluarSemantica(info, categorias)
        print(acabado)

        anomalias = acabado.filter((pl.col("anomalia_numerica") == 1) | (pl.col("anomalia_semantica") == True))
        exportarArchivos(Categorias(anomalias))
    except Exception as e:
        print(f"Error: {str(e)}")