from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
# Create your views here.

def home(request):
    return HttpResponse("<h1>¡Servidor Django activo!</h1><p>Esta es la API NFC. Usa <code>/api/nfc/</code> con POST para enviar datos.</p>")


@csrf_exempt
def receive_nfc_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            tag_id = data.get('tag_id')
            mensaje = data.get('mensaje')

            print(f"NFC recibido: {tag_id} - {mensaje}")

            return JsonResponse({'status': 'ok', 'message': 'Datos recibidos correctamente'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)