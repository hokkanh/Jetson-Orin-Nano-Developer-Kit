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
        Ottaa sisään kuvan (Numpy-matriisi), SUODATTAA KOHINAN, 
        hylkää liian pienet roskat ja palauttaa listan todellisista esteistä.
        """
        # --- 1. ESIKÄSITTELY JA SIGNAALINKÄSITTELY ---
        
        # Tarkistetaan, onko kuva 1-kanavainen (syvyys/harmaasävy)
        if len(kuva_2d.shape) == 2:
            # Muutetaan mahdollinen 16-bittinen syvyysdata 8-bittiseksi käsittelyä varten
            if kuva_2d.dtype == np.uint16 or kuva_2d.dtype == '<u2':
                kasiteltava_kuva = np.clip(kuva_2d * (255.0 / 8000.0), 0, 255).astype(np.uint8)
            else:
                kasiteltava_kuva = kuva_2d.astype(np.uint8).copy()

            # Ase 1: Tuhoaa mustat kuopat ja yksittäiset virhepikselit
            suodatettu = cv2.medianBlur(kasiteltava_kuva, 5)

            # Ase 2: Tuhoaa pienet roskat ja pöllyävät lehdet (Morfologinen avaus)
            kernel = np.ones((5, 5), np.uint8)
            suodatettu = cv2.morphologyEx(suodatettu, cv2.MORPH_OPEN, kernel)

            # Muutetaan suodatettu kuva YOLOa varten 3-kanavaiseksi (RGB)
            kuva_rgb = cv2.cvtColor(suodatettu, cv2.COLOR_GRAY2RGB)
        else:
            # Jos kuva on jo valmiiksi värikuva (RGB)
            kuva_rgb = kuva_2d.copy()

        # --- 2. TEKOÄLYN PÄÄTTELY ---
        # verbose=False pitää terminaalin puhtaana turhasta spämmitulosteesta
        results = self.model(kuva_rgb, verbose=False, conf=0.5)
        
        # --- 3. TULOSTEN PAKETOINTI JA OHJELMALLINEN FILTTERI ---
        loydetyt_kohteet = []
        
        for r in results:
            for box in r.boxes:
                # YOLO palauttaa xywh = x_center, y_center, width, height
                koordinaatit = box.xywh[0].tolist()
                leveys = int(koordinaatit[2])
                korkeus = int(koordinaatit[3])
                
                # OHJELMALLINEN FILTTERI: Hylätään fyysisesti liian pienet laatikot
                # Jos bounding box on alle 15x15 pikseliä, se on joko roska tai liian kaukana
                if leveys < 15 or korkeus < 15:
                    continue
                
                luokka_id = int(box.cls[0].item())
                varmuus = float(box.conf[0].item())
                
                # Rakennetaan helppolukuinen sanakirja jokaisesta löydetystä kohteesta
                loydetyt_kohteet.append({
                    "luokka": luokka_id,
                    "varmuus": varmuus,
                    "x": int(koordinaatit[0]),
                    "y": int(koordinaatit[1]),
                    "w": leveys,
                    "h": korkeus
                })
                
        return loydetyt_kohteet