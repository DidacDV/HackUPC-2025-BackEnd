import polars as pl
import datetime as dt
import os
import numpy as np
from pathlib import Path
from unidecode import unidecode
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
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
    return (dinero, tipoTransaccion, categoria, mismaDivisa, infoPagador, frecuenciaTemporal, motivo)

def guardarEnCSV(path, tuple, nombreArchivo):
    dinero, tipoTransaccion, categoria, mismaDivisa, infoPagador, frecuenciaTemporal, motivo = tuple

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
        return "Transacción normal"
    
    # Cargar modelo de lenguaje (puedes usar uno más pequeño si es necesario)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Crear un texto descriptivo de la transacción
    transaction_details = (
        f"Transacción con importe original {row['importe_orig.']} y importe pagado {row['importe_pagado']}. "
        f"Comisión: {row['comision']}, Tasa impuestos: {row['tasa_impuestos']}, "
        f"Ingreso: {row['ingreso']}, Gasto: {row['gasto']}. "
        f"Score de anomalía: {row['score_anomalia']:.2f}"
    )
    
    # Generar un embedding del texto
    embedding = model.encode(transaction_details)
    
    # Definir posibles razones de anomalías (podrías expandir esto)
    common_reasons = [
        "El importe pagado es significativamente diferente al importe original",
        "La comisión es inusualmente alta para este tipo de transacción",
        "La tasa de impuestos no coincide con los valores esperados",
        "La relación entre ingreso y gasto es atípica",
        "La duración de la transacción es anómala",
        "Patrón de transacción inusual en comparación con el comportamiento histórico",
        "El descuento aplicado es inesperadamente alto o bajo",
        "La comisión varía drásticamente sin una explicación lógica",
        "El importe de los impuestos es inconsistente con la tasa aplicada",
        "Se observa un redondeo inusual en los importes o comisiones",
        "Un ingreso o gasto es registrado fuera del rango esperado para la hora o fecha",
        "La proporción de gasto en una transacción específica es excesivamente alta o baja en comparación con el ingreso generado",
        "Se registran ingresos o gastos con valores idénticos de forma repetida",
        "Transacciones con importes similares presentan duraciones extremadamente diferentes",
        "La duración de la transacción es inusualmente corta o larga para el tipo de producto o servicio",
        "Un aumento o disminución repentino y significativo en el valor promedio de las transacciones",
        "Una frecuencia inusual de transacciones con valores específicos (picos o valles inesperados)",
        "Cambios drásticos en la distribución de los valores de las columnas numéricas"
    ]
    
    # Encontrar la razón más similar
    reason_embeddings = model.encode(common_reasons)
    similarities = cosine_similarity([embedding], reason_embeddings)
    best_match_idx = np.argmax(similarities)
    best_match = common_reasons[best_match_idx]
    
    # Construir el mensaje final
    explanation = (
        f"POSIBLE ANOMALÍA DETECTADA (score: {row['score_anomalia']:.2f}). "
        f"Razón más probable: {best_match}. "
        f"Detalles: {transaction_details}"
    )
    
    return explanation

def evaluarSemantica(data, categorias):
    use_embeddings = False
    
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # Alternativas ligeras:
        # - 'paraphrase-MiniLM-L3-v2' - muy ligero (33MB)
        # - 'distiluse-base-multilingual-cased-v1' - multilingüe
        # - 'paraphrase-multilingual-MiniLM-L12-v2' - buen balance
        use_embeddings = True
        print(f"Usando modelo de embeddings: {model.get_sentence_embedding_dimension()}d")
    except Exception as e:
        print(f"No se pudo cargar modelo de embeddings: {e}")
        print("Usando TF-IDF como alternativa...")
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            stop_words=['el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'de', 'del', 'en', 'por', 'para']
        )

    if use_embeddings:
        category_embeddings = {
            cat: model.encode(cat) for cat in categorias
        }
    else:
        corpus = []
        corpus_categories = []
        
        for row in data.rows(named=True):
            corpus.append(row["descripcion_gasto"])
            corpus_categories.append(row["categoria"])
        
        for cat in categorias:
            corpus.append(cat)
            corpus_categories.append(cat)
        
        tfidf_matrix = vectorizer.fit_transform(corpus)
    
    coherence_scores = []
    explanations = []

    for idx in range(data.height):
        row = data.row(idx, named=True)
        categoria = row["categoria"]
        descripcion = row["descripcion_gasto"]
        
        if not categoria or not descripcion:
            coherence_scores.append(0.0)
            explanations.append("categoria_o_descripcion_vacia")
            continue
        
        # Calcular similitud semántica
        if use_embeddings:
            try:
                # Codificar descripción y calcular similitud con la categoría
                desc_embedding = model.encode(descripcion)
                cat_embedding = category_embeddings[categoria]
                
                similarity = cosine_similarity([desc_embedding], [cat_embedding])[0][0]
                explanation = f"Similitud de embeddings: {similarity:.4f}"
            except Exception as e:
                similarity = 0.0
                explanation = f"Error en embeddings: {str(e)}"
        else:
            try:
                desc_idx = corpus.index(descripcion)
                cat_indices = [i for i, cat in enumerate(corpus_categories) if cat == categoria]
                
                # Calcular similitud promedio con todas las instancias de esta categoría i 
                desc_vector = tfidf_matrix[desc_idx]
                similarities = []
                
                for cat_idx in cat_indices:
                    cat_vector = tfidf_matrix[cat_idx]
                    sim = cosine_similarity(desc_vector, cat_vector)[0][0]
                    similarities.append(sim)
                
                similarity = np.mean(similarities) if similarities else 0.0
                explanation = f"Similitud TF-IDF: {similarity:.4f}"
            except Exception as e:
                similarity = 0.0
                explanation = f"Error en TF-IDF: {str(e)}"
        
        coherence_scores.append(float(similarity))
        explanations.append(explanation)
    
    data = data.with_columns([
        pl.Series(name="coherencia_semantica", values=coherence_scores),
        pl.Series(name="explicacion_semantica", values=explanations)
    ])
    
    # Definir umbral adaptativo para anomalías semánticas
    # Podemos usar un percentil bajo o un umbral fijo
    umbral = 0.05  # umbral fijo para considerar algo como anomalía semántica
    
    data = data.with_columns([
        (pl.col("coherencia_semantica") < umbral).alias("anomalia_semantica")
    ])
    
    return data

if __name__ == "__main__":
    
    try:
        path = get_Path("E1", "csv")
        data = pl.read_csv(path, try_parse_dates=True)

        final, categorias = crearNuevosDataSets(data)
        info = IsolationAnalisis(final)
        acabado = evaluarSemantica(info, categorias)


        anomalias = acabado.filter((pl.col("anomalia_numerica") == 1) | (pl.col("anomalia_semantica") == True))
        print(anomalias["explicacion_anomalia"])
        exportarArchivos(Categorias(info.filter(pl.col("anomalia_numerica") == 1))) 

    except Exception as e:
        print(f"Error: {str(e)}")