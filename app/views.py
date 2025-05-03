from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from web3 import Web3
import json, os
# Create your views here.


w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
contract_address = Web3.to_checksum_address("0x5fbdb2315678afecb367f032d93f642f64180aa3")  # dirección de tu XCOIN
abi_path = os.path.join(os.path.dirname(__file__), "XCOIN.abi.json")
with open(abi_path) as f:
    abi = json.load(f)["abi"]

@api_view(["GET"])
def balance_xcoin(request):
    address = request.GET.get("address")
    if not address:
        return Response({"error": "Missing 'address' parameter"}, status=400)

    try:
        checksum_address = Web3.to_checksum_address(address)
        balance = xcoin.functions.balanceOf(checksum_address).call()
        return Response({
            "address": checksum_address,
            "balance": balance
        })
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
def transfer_xcoin(request):
    print("📦 RAW BODY:", request.body)
    print("📊 request.data:", request.data)

    sender = request.data.get("from")
    recipient = request.data.get("to")
    amount = int(request.data.get("amount", 0))
    private_key = request.data.get("private_key")

    if not all([sender, recipient, amount, private_key]):
        return Response({"error": "Missing parameters"}, status=400)

    try:
        sender = Web3.to_checksum_address(sender)
        recipient = Web3.to_checksum_address(recipient)
        nonce = w3.eth.get_transaction_count(sender)

        tx = xcoin.functions.transfer(recipient, amount).build_transaction({
            'from': sender,
            'nonce': nonce,
            'gas': 200000,
            'gasPrice': w3.to_wei('5', 'gwei')
        })

        signed_tx = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        return Response({
            "status": "success",
            "tx": tx_hash.hex()
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


xcoin = w3.eth.contract(address=contract_address, abi=abi)


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