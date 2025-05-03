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

    return (dinero, tiempoTransaccion, tipoTransaccion, categoria, mismaDivisa, infoPagador, frecuenciaTemporal)

def guardarEnCSV(path, tuple, nombreArchivo):
    dinero, tipoTransaccion, categoria, mismaDivisa, infoPagador, frecuenciaTemporal = tuple

    os.makedirs(path, exist_ok=True)

    dinero.write_csv(f"{path}/{nombreArchivo}Dinero.csv")
    tipoTransaccion.write_csv(f"{path}/{nombreArchivo}tipo_transaccion.csv")
    categoria.write_csv(f"{path}/{nombreArchivo}categoria.csv")
    mismaDivisa.write_csv(f"{path}/{nombreArchivo}misma_divisa.csv")
    infoPagador.write_csv(f"{path}/{nombreArchivo}info_pagador.csv")

    frecuenciaTemporal[0].write_csv(f"{path}/{nombreArchivo}frecuencia_dia.csv")
    frecuenciaTemporal[1].write_csv(f"{path}/{nombreArchivo}frecuencia_mes.csv")
    frecuenciaTemporal[2].write_csv(f"{path}/{nombreArchivo}frecuencia_year.csv")

def exportarArchivos(tuple):
    current_dir = Path(__file__).parent
    csv_dir = current_dir / 'DataSets'
    guardarEnCSV(csv_dir, tuple, "E1")

def crearNuevosDataSets(base):
    tipoCorrecto = ProcesamientoInicial(base)
    setLimpaido = limpiarDataSet(tipoCorrecto)
    categorias = set(setLimpaido["categoria"].unique().to_list())
    #NuevaData = Categorias(setLimpaido)    
    
    
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

    return df

def evaluarSemantica(data, categorias):
    use_embeddings = False
    
    try:
        # Intentar usar un modelo de embeddings ligero
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
        
        # Crear vectorizador TF-IDF como fallback
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
        print(categorias)
        #acabado = evaluarSemantica(final, categorias)
        #print(acabado.filter(pl.col("anomalia_semantica") == True))
        #print(info.filter(pl.col("anomalia_numerica") == 1))
        #exportarArchivos(final)
    except Exception as e:
        print(f"Error: {str(e)}")