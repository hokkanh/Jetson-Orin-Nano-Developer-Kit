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
            # Koitetaan ladata TensorRT Jetsonia varten
            self.model = YOLO('yolov8n.engine', task='detect')
            logger.info("TensorRT-malli (.engine) ladattu onnistuneesti!")
        except Exception:
            # Fallback paikalliseen testaukseen
            self.model = YOLO(mallin_polku) 
            logger.warning(f"Ei TensorRT-mallia. Käytetään PyTorch-mallia ({mallin_polku}) fallbackina.")

    def etsi_esteet(self, kuva_2d):
        """
        Ottaa sisään kuvan (Numpy-matriisi) ja palauttaa listan löydetyistä esteistä.
        """
        # 1. YOLO VAATII 3 VÄRIKANAVAA (RGB).
        # Tarkistetaan, onko datalukijalta saatu kuva 1-kanavainen (syvyys/harmaasävy).
        if len(kuva_2d.shape) == 2:
            # Jos data on 16-bittistä syvyysdataa, se pitää puristaa 8-bittiseksi.
            if kuva_2d.dtype == np.uint16 or kuva_2d.dtype == '<u2':
                skaalattu = np.clip(kuva_2d * (255.0 / 8000.0), 0, 255).astype(np.uint8)
                kuva_rgb = cv2.cvtColor(skaalattu, cv2.COLOR_GRAY2RGB)
            else:
                # Jos se on jo 8-bittistä harmaasävyä, muutetaan vain suoraan feikki-RGB:ksi
                kuva_rgb = cv2.cvtColor(kuva_2d.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            # Kuva oli jo valmiiksi RGB
            kuva_rgb = kuva_2d

        # 2. AJETAAN KUVA TEKOÄLYN LÄPI
        # verbose=False pitää terminaalin puhtaana turhasta spämmitulosteesta
        results = self.model(kuva_rgb, verbose=False)
        
        # 3. PAKETOIDAAN TULOKSET MAIN.PY:TÄ VARTEN
        loydetyt_kohteet = []
        
        for r in results:
            for box in r.boxes:
                # YOLO palauttaa xywh = x_center, y_center, width, height
                koordinaatit = box.xywh[0].tolist()
                luokka_id = int(box.cls[0].item())
                varmuus = float(box.conf[0].item())
                
                # Rakennetaan helppolukuinen sanakirja jokaisesta löydetystä kohteesta
                loydetyt_kohteet.append({
                    "luokka": luokka_id,
                    "varmuus": varmuus,
                    "x": int(koordinaatit[0]),
                    "y": int(koordinaatit[1]),
                    "w": int(koordinaatit[2]),
                    "h": int(koordinaatit[3])
                })
                
        return loydetyt_kohteet