from ultralytics import YOLO
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self):
        # Ladataan malli heti kun luokka alustetaan
        try:
            self.model = YOLO('yolov8n.engine', task='detect')
            logger.info("TensorRT-malli ladattu onnistuneesti!")
        except Exception:
            self.model = YOLO('yolov8n.pt') 
            logger.warning("Ei TensorRT-mallia. Käytetään PyTorch-mallia fallbackina.")

    def etsi_esteet(self, kuva_2d):
        """
        Tämä on funktio, jota main.py kutsuu joka ikisellä kuvaruudulla!
        Ottaa sisään Numpyn 2D-matriisin, palauttaa listan löydetyistä kohteista.
        """
        # 1. YOLO vaatii 3 värikanavaa (RGB). 
        # Tarkistetaan onko saamamme kuva 1-kanavainen (syvyys/harmaasävy)
        if len(kuva_2d.shape) == 2:
            # Kova Labsin syvyysdata saattaa olla 16-bittistä. Se pitää puristaa 8-bittiseksi YOLOa varten.
            if kuva_2d.dtype == np.uint16 or kuva_2d.dtype == '<u2':
                skaalattu = np.clip(kuva_2d * (255.0 / 8000.0), 0, 255).astype(np.uint8)
                kuva_rgb = cv2.cvtColor(skaalattu, cv2.COLOR_GRAY2RGB)
            else:
                kuva_rgb = cv2.cvtColor(kuva_2d, cv2.COLOR_GRAY2RGB)
        else:
            kuva_rgb = kuva_2d

        # 2. Ajetaan kuva tekoälyn läpi (verbose=False pitää konsolin siistinä)
        results = self.model(kuva_rgb, verbose=False)
        
        # 3. Paketoidaan tulokset siistiin listaan
        loydetyt_kohteet = []
        
        for r in results:
            for box in r.boxes:
                # xywh = x_center, y_center, width, height
                koordinaatit = box.xywh[0].tolist()
                luokka_id = int(box.cls[0].item())
                varmuus = float(box.conf[0].item())
                
                loydetyt_kohteet.append({
                    "luokka": luokka_id,
                    "varmuus": varmuus,
                    "x": int(koordinaatit[0]),
                    "y": int(koordinaatit[1]),
                    "w": int(koordinaatit[2]),
                    "h": int(koordinaatit[3])
                })
                
        return loydetyt_kohteet