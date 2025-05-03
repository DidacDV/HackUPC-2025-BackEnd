import csv
import os
from django.http import JsonResponse, Http404
from django.conf import settings

ARCHIVOS_VALIDOS = {
    'anomalia_categorias': 'E1categoria.csv',
    'anomalia_gastos_ingresos': 'E1Dinero.csv',
    'anomalia_dia_transacciones': 'E1frecuencia_dia.csv',
    'anomalia_mes_transacciones': 'E1frecuencia_mes.csv',
    'anomalia_año_transacciones': 'E1frecuencia_year.csv',
    'anomalia_info_cuenta': 'E1info_pagador.csv',
    'anomalia_misma_divisa': 'E1misma_divisa.csv',
    'anomalia_descripcion_transaccion': 'E1tipo_transaccion.csv',
}

def mostrar_csv_dinamico(request, nombre_csv):
    if nombre_csv not in ARCHIVOS_VALIDOS:
        raise Http404("Archivo no permitido")

    nombre_archivo = ARCHIVOS_VALIDOS[nombre_csv]
    ruta_archivo = os.path.join(settings.BASE_DIR, 'app', 'static', nombre_archivo)
    
    datos = []
    try:
        with open(ruta_archivo, newline='', encoding='utf-8') as csvfile:
            lector = csv.DictReader(csvfile)
            for fila in lector:
                datos.append(fila)
        return JsonResponse(datos, safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
