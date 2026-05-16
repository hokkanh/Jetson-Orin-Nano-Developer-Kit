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
        # Kova Labsin käyttämä topic syvyyskameralle
        self.topic = "/camera/camera/infra1/image_rect_raw"
        
        if kohdetiedosto:
            self.mcap_files = [self.data_dir / kohdetiedosto]
            logger.info(f"Asetettu lukemaan VAIN tiedosto: {kohdetiedosto}")
        else:
            self.mcap_files = sorted(list(self.data_dir.glob("*.mcap")))
            logger.info(f"Löydettiin {len(self.mcap_files)} tiedostoa.")

    def lue_kuvat_generaattorina(self):
        """
        Lypsää mcap-tiedostosta yhden kuvan kerrallaan (yield)
        ilman että tietokoneen muisti räjähtää.
        """
        for tiedosto in self.mcap_files:
            logger.info(f"Aloitetaan tiedoston purku: {tiedosto.name}")
            
            try:
                # Avataan mcap-tiedosto lukijalla
                iterator = read_ros2_messages(str(tiedosto), topics=[self.topic])
                
                for viesti in iterator:
                    img_msg = viesti.ros_msg
                    raakadata = bytes(img_msg.data)
                    
                    # Muutetaan raakadata Numpyn ymmärtämään muotoon (16-bit)
                    matriisi = np.frombuffer(raakadata, dtype=np.uint8)
                    
                    # Muotoillaan litteä pötkö takaisin 2D-kuvaksi
                    kuva_2d = matriisi.reshape((img_msg.height, img_msg.width))
                    
                    # Palautetaan yksi kuva kerrallaan main.py:lle
                    yield kuva_2d
                    
            except Exception as e:
                logger.error(f"Tiedosto {tiedosto.name} aiheutti virheen: {e}")
                continue