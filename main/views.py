from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import torch
from torchvision import transforms
import json
import numpy as np
from . import thyronetmodel


@csrf_exempt
def handleData(request):
    if request.method == 'POST':
        # Retrieve basic form data
        name = request.POST.get('name')
        age = request.POST.get('age')
        sex = request.POST.get('gender')

        toggle_scan = request.POST.get('toggleScanDiagnosis')  # Ultrasound Scan
        toggle_blood = request.POST.get('toggleBloodDiagnosis')  # Blood Test

        # --- Process Ultrasound Scan (CNN-ViT Model) ---
        scan_image = request.FILES.get('scanImage')
        image_data = None
        if scan_image:
            try:
                image = Image.open(scan_image).convert("RGB")
                image_data = image  # Store the image object
            except Exception as e:
                return JsonResponse({'error': f'Invalid image file: {str(e)}'}, status=400)
        
        if toggle_scan and image_data is not None:
            try:
                scan_target, scan_target_summary = thyronetmodel.predict_tirads(image_data)
                scan_result = {
                    'target' : scan_target,
                    'target_summary' : scan_target_summary,
                }
            except Exception as e:
                scan_result = {"error": str(e)}
        else:
            scan_result = {
                "target" : None,
                "target_summary" : None
                }

        # --- Process Blood Test (XGBoost Model) ---
        # Retrieve blood test values; if an input is empty, set it to NaN.
        tsh = request.POST.get('tsh')
        freeT4 = request.POST.get('freeT4')
        freeT3 = request.POST.get('freeT3')
        totalT4 = request.POST.get('totalT4')
        antiTpo = request.POST.get('antiTpo')
        antiTg = request.POST.get('antiTg')

        if toggle_blood:
            try:
                # Assemble input for the XGBoost model.
                # Note: The blood test model expects keys with specific names. Here, we default "Sex" to "Female" (adjust if needed).
                new_input = {
                    "Age": float(age) if age and age != '' else np.nan,
                    "Sex": sex,
                    "TSH": float(tsh) if tsh and tsh != '' else np.nan,
                    "Free T4": float(freeT4) if freeT4 and freeT4 != '' else np.nan,
                    "Free T3": float(freeT3) if freeT3 and freeT3 != '' else np.nan,
                    "Total T4": float(totalT4) if totalT4 and totalT4 != '' else np.nan,
                    "AntiTPO": float(antiTpo) if antiTpo and antiTpo != '' else np.nan,
                    "AntiTg": float(antiTg) if antiTg and antiTg != '' else np.nan,
                }
                predicted_condition, class_probabilities = thyronetmodel.predict_condition(new_input)
                blood_result = {
                    "target": predicted_condition,
                    "target_summary": class_probabilities # Convert JSON string to a dict
                }
            except Exception as e:
                blood_result = {"error": str(e)}
        else:
            blood_result = {
                "target" : None,
                "target_summary" : None
                }

        data = {
            'message': 'OK',
            'scan_result': scan_result,
            'blood_result': blood_result
        }
        print(json.dumps(data, indent=4))
        return JsonResponse({'data': data}, status=200)
    
    return JsonResponse({'SERVER STATUS': 'ONLINE'}, status=400)