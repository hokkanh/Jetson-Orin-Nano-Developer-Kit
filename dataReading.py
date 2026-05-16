from pathlib import Path
import numpy as np
import logging
from mcap_ros2.reader import read_ros2_messages

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MCAPReader:
    # KORJATTU: __init__ vaatii alaviivat, jotta Python tunnistaa tämän!
    def __init__(self, data_dir: str = "data/mcap_files", kohdetiedosto: str = None, topic: str = "/camera/camera/infra1/image_rect_raw"):
        self.data_dir = Path(data_dir)
        self.topic = topic
        
        if kohdetiedosto:
            self.mcap_files = [self.data_dir / kohdetiedosto]
            logger.info(f"Asetettu lukemaan VAIN tiedosto: {kohdetiedosto}")
        else:
            self.mcap_files = sorted(list(self.data_dir.glob("*.mcap")))

    def lue_kuvat_generaattorina(self):
        for tiedosto in self.mcap_files:
            logger.info(f"Luetaan kanavaa {self.topic} tiedostosta: {tiedosto.name}")
            try:
                iterator = read_ros2_messages(str(tiedosto), topics=[self.topic])
                
                for viesti in iterator:
                    img_msg = viesti.ros_msg
                    raakadata = bytes(img_msg.data)
                    
                    # AUTOMAATTINEN DATATYYPIN TUNNISTUS
                    # 16UC1 tarkoittaa 16-bittistä syvyysdataa. Mono8 on 8-bit infrapunaa.
                    if img_msg.encoding == "16UC1":
                        matriisi = np.frombuffer(raakadata, dtype="<u2")
                    else:
                        matriisi = np.frombuffer(raakadata, dtype=np.uint8)
                    
                    kuva_2d = matriisi.reshape((img_msg.height, img_msg.width))
                    yield kuva_2d
                    
            except Exception as e:
                logger.error(f"Virhe tiedostossa {tiedosto.name}: {e}")
                continue