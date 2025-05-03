import polars as pl # type: ignore
import datetime as dt
import os
from pathlib import Path
from unidecode import unidecode

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
    df = data.with_columns([
        pl.col("importe_orig.").cast(pl.Float64, strict=False),
        pl.col("importe_pagado").cast(pl.Float64, strict=False),
        pl.col("comision").cast(pl.Float64, strict=False),
        pl.col("tasa_impuestos").cast(pl.Float64, strict=False),
        pl.col("imp._impuestos").cast(pl.Float64, strict=False)
    ])
    return df


def Categorias(data):
    #Caracterizacion

    dinero = data.select([pl.col("id_transaccion"), pl.col("ingreso"), pl.col("gasto"), pl.col("id_gasto")])
    tipoTransaccion = data.select([pl.col("id_transaccion"), pl.col("tipo"), pl.col("descripcion"), pl.col("descripcion_gasto")])
    categoria = data.select([pl.col("id_transaccion"), pl.col("categoria"), pl.col("codigo_categoria")])
    mismaDivisa = data.select([pl.col("id_transaccion"), (pl.col("divisa_original") == pl.col("moneda_de_pago")).alias("misma_divisa")])
    infoPagador = data.select([pl.col("id_transaccion"), pl.col("nn_tarjeta"), pl.col("pagador")])
    frecuenciaTemporal = (data.with_columns([pl.col("hora_inicio_utc").dt.date().alias("dia")]).group_by("dia").agg([pl.count().alias("n_transacciones")]).sort("dia"),
                          data.with_columns([pl.col("hora_inicio_utc").dt.strftime("%Y-%m").alias("mes")]).group_by("mes").agg([pl.count().alias("n_transacciones")]).sort("mes"),
                          data.with_columns([pl.col("hora_inicio_utc").dt.strftime("%Y").alias("año")]).group_by("año").agg([pl.count().alias("n_transacciones")]).sort("año")
                        )

    return (dinero, tipoTransaccion, categoria, mismaDivisa, infoPagador, frecuenciaTemporal)

def guardarEnCSV(path, tuple, nombreArchivo):
    dinero, tipoTransaccion, categoria, mismaDivisa, infoPagador, frecuenciaTemporal = tuple

    os.makedirs(path, exist_ok=True)

    dinero.write_csv(f"{path}/{nombreArchivo}Dinero.csv")
    tipoTransaccion.write_csv(f"{path}/{nombreArchivo}tipo_transaccion.csv")
    categoria.write_csv(f"{path}/{nombreArchivo}categoria.csv")
    mismaDivisa.write_csv(f"{path}/{nombreArchivo}misma_divisa.csv")
    infoPagador.write_csv(f"{path}/{nombreArchivo}info_pagador.csv")

    frecuenciaTemporal[0].write_csv(f"{path}/{nombreArchivo}frecuencia_por_dia.csv")
    frecuenciaTemporal[1].write_csv(f"{path}/{nombreArchivo}frecuencia_por_mes.csv")
    frecuenciaTemporal[2].write_csv(f"{path}/{nombreArchivo}frecuencia_por_year.csv")

def crearNuevosDataSets(base):
    tipoCorrecto = ProcesamientoInicial(base)
    NuevaData = Categorias(tipoCorrecto)    
    return NuevaData

if __name__ == "__main__":
    
    try:
        path = get_Path("E1", "csv")
        data = pl.read_csv(path, try_parse_dates=True)
        data.columns = [
            unidecode(col.strip().lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("º", "n")) for col in data.columns]        
        final = crearNuevosDataSets(data)
        current_dir = Path(__file__).parent
        csv_dir = current_dir / 'DataSets'
        guardarEnCSV(csv_dir, final, "E1")
    except Exception as e:
        print(f"Error: {str(e)}")