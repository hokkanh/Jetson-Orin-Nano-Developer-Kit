import cv2
import numpy as np
import logging
from ultralytics import YOLO

# Asetetaan lokitus, jotta näemme mitä mallille tapahtuu
logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, mallin_polku: str = 'yolov8n.pt'):
        """
        Alustaa tekoälyn. Yrittää ensin Kova Labsin vaatimaa .engine -tiedostoa.
        Jos sitä ei löydy (esim. kun koodaat Macilla), lataa standardin .pt -mallin.
        """
        try:
            self.model = YOLO('yolov8n.engine', task='detect')
            logger.info("TensorRT-malli (.engine) ladattu onnistuneesti!")
        except Exception:
            self.model = YOLO(mallin_polku) 
            logger.warning(f"Ei TensorRT-mallia. Käytetään PyTorch-mallia ({mallin_polku}) fallbackina.")

    def etsi_esteet(self, kuva_2d):
        """
        Ottaa sisään kuvan (Numpy-matriisi), SUODATTAA VARJOT POIS fysiikalla, 
        hylkää pienet roskat ja palauttaa listan todellisista esteistä.
        """
        # --- 1. ESIKÄSITTELY JA SIGNAALINKÄSITTELY ---
        
        if len(kuva_2d.shape) == 2:
            kasiteltava_kuva = kuva_2d.astype(np.uint8).copy()

            # ASE 1: CLAHE - Nostetaan oikeat kohteet esiin taustasta
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            kontrasti_kuva = clahe.apply(kasiteltava_kuva)

            # ASE 2: VARJON TAPPAJA (Threshold - Kynnystys)
            # THRESH_TOZERO tarkoittaa: "Jos pikselin arvo on alle 40 (musta varjo), muuta se täysin mustaksi (0)."
            # Voit joutua säätämään tätä numeroa 40 joko ylös (esim 60) tai alas (esim 20)
            _, suodatettu = cv2.threshold(kontrasti_kuva, 40, 255, cv2.THRESH_TOZERO)

            # ASE 3: Roskien siivous (Mediaanifiltteri ja Morfologia)
            suodatettu = cv2.medianBlur(suodatettu, 5)
            kernel = np.ones((5, 5), np.uint8)
            suodatettu = cv2.morphologyEx(suodatettu, cv2.MORPH_OPEN, kernel)

            # Muutetaan YOLOa varten väriksi
            kuva_rgb = cv2.cvtColor(suodatettu, cv2.COLOR_GRAY2RGB)
        else:
            kuva_rgb = kuva_2d.copy()

        # --- 2. TEKOÄLYN PÄÄTTELY ---
        # Laskettiin conf-arvoa takaisin hieman, koska varjot tapetaan nyt yläpuolella!
        results = self.model(kuva_rgb, verbose=False, conf=0.35)
        
        # --- 3. TULOSTEN PAKETOINTI JA OHJELMALLINEN FILTTERI ---
        loydetyt_kohteet = []
        
        for r in results:
            for box in r.boxes:
                koordinaatit = box.xywh[0].tolist()
                leveys = int(koordinaatit[2])
                korkeus = int(koordinaatit[3])
                
                # OHJELMALLINEN FILTTERI: Hylätään fyysisesti liian pienet (alle 15x15)
                if leveys < 15 or korkeus < 15:
                    continue
                
                luokka_id = int(box.cls[0].item())
                varmuus = float(box.conf[0].item())
                
                loydetyt_kohteet.append({
                    "luokka": luokka_id,
                    "varmuus": varmuus,
                    "x": int(koordinaatit[0]),
                    "y": int(koordinaatit[1]),
                    "w": leveys,
                    "h": korkeus
                })
                
        return loydetyt_kohteet