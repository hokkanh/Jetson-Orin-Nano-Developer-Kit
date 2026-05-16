from pathlib import Path
import numpy as np
import logging
from mcap_ros2.reader import read_ros2_messages

# Asetetaan yksinkertainen lokitus
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MCAPReader:
    def __init__(self, data_dir: str = "data/mcap_files", kohdetiedosto: str = None):
        self.data_dir = Path(data_dir)
        
        # TÄMÄ ON KRIITTINEN! (Kts. selitys alla)
        self.topic = "/camera/camera/infra1/image_rect_raw"
        
        if kohdetiedosto:
            # Jos pyydetään vain yhtä (esim. "dualtarget.mcap"), ladataan vain se!
            self.mcap_files = [self.data_dir / kohdetiedosto]
            logger.info(f"Asetettu lukemaan VAIN tiedosto: {kohdetiedosto}")
        else:
            # Muuten etsitään ne kaikki 4 aakkosjärjestyksessä
            self.mcap_files = sorted(list(self.data_dir.glob("*.mcap")))
            logger.info(f"Löydettiin {len(self.mcap_files)} tiedostoa: {[f.name for f in self.mcap_files]}")
            
    def lue_kuvat_generaattorina(self):
        """
        TÄMÄ ON TÄRKEIN METODI. 
        Käyttää 'yield' -komentoa, joten muistia kuluu vain yhden kuvan verran (n. 1-2 MB),
        vaikka mcap-tiedostoja olisi 100 gigatavua.
        """
        for tiedosto in self.mcap_files:
            logger.info(f"Aloitetaan tiedoston purku: {tiedosto.name}")
            
            try:
                # Avataan mcap-tiedosto lukijalla
                iterator = read_ros2_messages(str(tiedosto), topics=[self.topic])
                
                for viesti in iterator:
                    # 1. Otetaan raakadata ulos viestistä
                    img_msg = viesti.ros_msg
                    raakadata = bytes(img_msg.data)
                    
                    # 2. Muutetaan raakadata Numpyn ymmärtämään muotoon
                    # Huom! Syvyyskameran data on 16-bittistä (uint16 eli "<u2")
                    # Jos luette tavallista värikuvaa, tämä pitää vaihtaa 8-bittiseksi (np.uint8)
                    matriisi = np.frombuffer(raakadata, dtype="<u2")
                    
                    # 3. Muotoillaan litteä pötkö takaisin 2D-kuvaksi (korkeus x leveys)
                    kuva_2d = matriisi.reshape((img_msg.height, img_msg.width))
                    
                    # 4. YIELD-TAIKA: Palautetaan yksi kuva kerrallaan pyytäjälle!
                    yield kuva_2d
                    
            except Exception as e:
                logger.error(f"Tiedosto {tiedosto.name} on korruptoitunut tai väärää tyyppiä: {e}")
                continue # Hypätään virheen yli seuraavaan tiedostoon