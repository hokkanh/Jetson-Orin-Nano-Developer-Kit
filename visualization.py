import cv2

class Visualizer:
    def __init__(self):
        pass

    def varita_syvyys(self, paikattu_kuva):
        """
        Ottaa harmaasävyisen syvyyskuvan ja muuttaa sen JET-lämpökartaksi.
        """
        return cv2.applyColorMap(paikattu_kuva, cv2.COLORMAP_JET)