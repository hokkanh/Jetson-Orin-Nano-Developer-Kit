from pathlib import Path
import numpy as np
import logging
from mcap_ros2.reader import read_ros2_messages

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MCAPReader:
    # 1. SIISTITTY: kohdetiedostolla on nyt oletusnimi, ja se otetaan suoraan käyttöön
    def __init__(self, data_dir: str = "data/mcap_files", kohdetiedosto: str = "dualtarget.mcap", topic: str = "/camera/camera/depth/image_rect_raw"):
        self.data_dir = Path(data_dir)
        self.topic = topic
        
        # Luodaan suoraan polku kyseiseen tiedostoon ilman listoja tai kikkailuja
        self.mcap_polku = self.data_dir / kohdetiedosto
        logger.info(f"Asetettu lukemaan tiedosto: {self.mcap_polku.name}")

    def lue_kuvat_generaattorina(self):
        logger.info(f"Luetaan kanavaa {self.topic} tiedostosta: {self.mcap_polku.name}")
        try:
            # 2. OPTIMOITU: Luetaan suoraan se yksi määrätty tiedosto ilman for-looppia
            iterator = read_ros2_messages(str(self.mcap_polku), topics=[self.topic])
            
            for viesti in iterator:
                img_msg = viesti.ros_msg
                raakadata = bytes(img_msg.data)
                
                # Tulkitaan data suoraan 16-bittiseksi syvyysmatriisiksi
                matriisi = np.frombuffer(raakadata, dtype="<u2")
                
                kuva_2d = matriisi.reshape((img_msg.height, img_msg.width))
                yield kuva_2d
                
        except Exception as e:
            logger.error(f"Virhe tiedoston {self.mcap_polku.name} lukemisessa: {e}")