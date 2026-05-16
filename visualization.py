import cv2
import numpy as np

class Visualizer:
    def __init__(self, ikkunan_nimi="Taktinen Näkymä (HUD)"):
        self.ikkunan_nimi = ikkunan_nimi
        
        # YOLOv8-mallin yleisimpiä luokkia (voit muokata Kova Labsin kisan mukaan!)
        self.luokat = {
            0: "Henkilo",
            1: "Polkupyoera",
            2: "Auto",
            3: "Moottoripyoera"
        }

    def piirra_hud(self, kuva_2d, esteet):
        """
        Ottaa sisään alkuperäisen matriisikuvan ja listan löydetyistä esteistä.
        Piirtää laatikot, tekstit ja näyttää tuloksen ruudulla.
        """
        # 1. VARMISTETAAN KUVAN VÄRIT (OpenCV käyttää BGR-muotoa)
        if len(kuva_2d.shape) == 2:
            # Muutetaan mahdollinen 16-bittinen syvyysdata 8-bittiseksi, jotta sen voi piirtää
            if kuva_2d.dtype == np.uint16 or kuva_2d.dtype == '<u2':
                skaalattu = np.clip(kuva_2d * (255.0 / 8000.0), 0, 255).astype(np.uint8)
                naytettava_kuva = cv2.cvtColor(skaalattu, cv2.COLOR_GRAY2BGR)
            else:
                naytettava_kuva = cv2.cvtColor(kuva_2d.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            naytettava_kuva = kuva_2d.copy()

        # 2. PIIRRETÄÄN JOKAINEN LÖYDETTY ESTE
        for este in esteet:
            # YOLO antaa keskipisteen ja koon, mutta cv2 haluaa ylä- ja alakulmat
            x_keski = este["x"]
            y_keski = este["y"]
            leveys = este["w"]
            korkeus = este["h"]

            # Lasketaan kulmat
            x_min = int(x_keski - (leveys / 2))
            y_min = int(y_keski - (korkeus / 2))
            x_max = int(x_keski + (leveys / 2))
            y_max = int(y_keski + (korkeus / 2))

            # Luodaan visuaalinen tyyli
            varmuus_prosentti = int(este["varmuus"] * 100)
            vari = (0, 255, 0) # Vihreä väri (B, G, R)
            paksuus = 2

            # Piirretään itse laatikko
            cv2.rectangle(naytettava_kuva, (x_min, y_min), (x_max, y_max), vari, paksuus)

            # Piirretään teksti laatikon päälle (esim. "Henkilo: 85%")
            luokan_nimi = self.luokat.get(este["luokka"], f"Kohde {este['luokka']}")
            teksti = f"{luokan_nimi} {varmuus_prosentti}%"
            
            cv2.putText(naytettava_kuva, teksti, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, vari, paksuus)

        # 3. NÄYTETÄÄN KUVA RUUDULLA
        cv2.imshow(self.ikkunan_nimi, naytettava_kuva)
        
        # Odotetaan 1 ms, jotta ikkuna ehtii päivittyä. 
        # Palauttaa painetun näppäimen, jotta ohjelman voi sulkea.
        return cv2.waitKey(1) & 0xFF