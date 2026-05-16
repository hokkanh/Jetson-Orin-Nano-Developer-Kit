import cv2
import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, mallin_polku: str = 'yolov8n.pt'):
        try:
            self.model = YOLO('yolov8n.engine', task='detect')
            logger.info("TensorRT-malli (.engine) ladattu onnistuneesti!")
        except Exception:
            self.model = YOLO(mallin_polku) 
            logger.warning(f"Ei TensorRT-mallia. Käytetään PyTorch-mallia ({mallin_polku}) fallbackina.")

    def etsi_esteet(self, kuva_2d):
        # 1. POISTETTU SIGNAALINKÄSITTELY: Palautetaan kuva luonnolliseksi
        if len(kuva_2d.shape) == 2:
            kasiteltava_kuva = kuva_2d.astype(np.uint8).copy()
            kuva_rgb = cv2.cvtColor(kasiteltava_kuva, cv2.COLOR_GRAY2RGB)
        else:
            kuva_rgb = kuva_2d.copy()

        # 2. TEKOÄLY: Laitetaan conf-arvo sopivaksi (esim 0.35)
        results = self.model(kuva_rgb, verbose=False, conf=0.35)
        
        # 3. GEOMETRINEN SUODATUS (Uusi älykäs filtteröinti)
        loydetyt_kohteet = []
        kuvan_korkeus = kuva_rgb.shape[0] # Yleensä 480
        
        # Määritetään kuollut kulma: esim. alimmat 130 pikseliä hylätään suoraan
        varjon_raja = kuvan_korkeus - 130 
        
        for r in results:
            for box in r.boxes:
                koordinaatit = box.xywh[0].tolist()
                x = int(koordinaatit[0])
                y = int(koordinaatit[1]) # Kohteen keskipisteen Y-koordinaatti
                w = int(koordinaatit[2])
                h = int(koordinaatit[3])
                
                # FILTTERI A: Hylätään mikroskooppiset roskat
                if w < 15 or h < 15:
                    continue
                    
                # FILTTERI B: VARJON TAPPAJA! 
                # Jos kohteen keskipiste on kuolleessa kulmassa (alhaalla), se on drone.
                if y > varjon_raja:
                    continue
                
                luokka_id = int(box.cls[0].item())
                varmuus = float(box.conf[0].item())
                
                loydetyt_kohteet.append({
                    "luokka": luokka_id,
                    "varmuus": varmuus,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })
                
        return loydetyt_kohteet