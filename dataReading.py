from pathlib import Path
import numpy as np
import logging
from mcap_ros2.reader import read_ros2_messages

logger = logging.getLogger(__name__)

class MCAPFusionReader:
    def __init__(self, data_dir: str = "data/mcap_files", kohdetiedosto: str = None):
        self.data_dir = Path(data_dir)
        self.infra_topic = "/camera/camera/infra1/image_rect_raw"
        self.depth_topic = "/camera/camera/depth/image_rect_raw"
        
        if kohdetiedosto:
            self.tiedosto = self.data_dir / kohdetiedosto
        else:
            self.tiedosto = sorted(list(self.data_dir.glob("*.mcap")))[0]
        logger.info(f"Yhdistetty tehokolukija alustettu tiedostolle: {self.tiedosto.name}")

    def lue_synkronoitu_virta(self):
        """
        Lukee SEKÄ infrapunan ETTA syvyyden yhdellä tiedoston läpikäynnillä.
        Estää kovalevyn tukkeutumisen ja säästää prosessoritilaa.
        """
        try:
            # Annetaan molemmat topicit listana samalle iteraattorille
            iterator = read_ros2_messages(str(self.tiedosto), topics=[self.infra_topic, self.depth_topic])
            
            for viesti in iterator:
                img_msg = viesti.ros_msg
                raakadata = bytes(img_msg.data)
                
                # Tunnistetaan kanava viestityypin perusteella
                if img_msg.encoding == "16UC1":
                    matriisi = np.frombuffer(raakadata, dtype="<u2")
                    tyyppi = "depth"
                else:
                    matriisi = np.frombuffer(raakadata, dtype=np.uint8)
                    tyyppi = "infra"
                
                kuva_2d = matriisi.reshape((img_msg.height, img_msg.width))
                yield tyyppi, kuva_2d
                
        except Exception as e:
            logger.error(f"Virhe MCAP-virran luvussa: {e}")