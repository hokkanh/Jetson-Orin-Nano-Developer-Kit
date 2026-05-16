import cv2
import numpy as np
import math

class Visualizer:
    def __init__(self, ikkunan_nimi="Taktinen HUD"):
        self.ikkunan_nimi = ikkunan_nimi
        self.luokat = {0: "Henkilo", 1: "Pyoera", 2: "Auto"}

    def piirra_hud(self, kuva_2d, esteet):
        # 1. VARMISTETAAN VÄRIT JA KOOT
        if len(kuva_2d.shape) == 2:
            if kuva_2d.dtype == np.uint16 or kuva_2d.dtype == '<u2':
                skaalattu = np.clip(kuva_2d * (255.0 / 8000.0), 0, 255).astype(np.uint8)
                naytettava_kuva = cv2.cvtColor(skaalattu, cv2.COLOR_GRAY2BGR)
            else:
                naytettava_kuva = cv2.cvtColor(kuva_2d.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            naytettava_kuva = kuva_2d.copy()

        korkeus, leveys = naytettava_kuva.shape[:2]
        keski_x = leveys // 2
        keski_y = korkeus // 2

        # --- OMINAISUUS 3: TAKTINEN TUTKA (Alustetaan tyhjä musta ruutu) ---
        tutka_koko = 180
        tutka = np.zeros((tutka_koko, tutka_koko, 3), dtype=np.uint8)
        
        # Piirretään meidän drone tutkan keskelle alareunaan
        drone_tutka_x = tutka_koko // 2
        drone_tutka_y = tutka_koko - 15
        cv2.circle(tutka, (drone_tutka_x, drone_tutka_y), 5, (255, 255, 255), -1)
        cv2.putText(tutka, "DRONE", (drone_tutka_x - 20, drone_tutka_y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # 2. KÄYDÄÄN LÄPI ESTEET JA TEHDÄÄN UHKA-ARVIOINTI
        for este in esteet:
            x_keski = este["x"]
            y_keski = este["y"]
            w = este["w"]
            h = este["h"]

            x_min = int(x_keski - (w / 2))
            y_min = int(y_keski - (h / 2))
            x_max = int(x_keski + (w / 2))
            y_max = int(y_keski + (h / 2))

            # --- OMINAISUUS 2: UHKA-ARVIOINTI (Threat Matrix) ---
            # Lasketaan kuinka kaukana kohde on ruudun keskipisteestä (lentoreitistä)
            etaisyys_keskelta = math.sqrt((x_keski - keski_x)**2 + (y_keski - keski_y)**2)
            
            # Jos etäisyys on pieni, kohde on aivan dronen edessä
            if etaisyys_keskelta < 160: 
                vari = (0, 0, 255) # Punainen = Vaara!
                teksti = "VAARA"
                paksuus = 3
                # Piirretään varoitusviiva ruudun keskeltä suoraan kohteeseen
                cv2.line(naytettava_kuva, (keski_x, keski_y), (x_keski, y_keski), vari, 2)
            else:
                vari = (0, 255, 0) # Vihreä = Turvallinen etäisyys
                teksti = "SEURANTA"
                paksuus = 2

            # Piirretään laatikko ja tekstit itse pääkuvaan
            cv2.rectangle(naytettava_kuva, (x_min, y_min), (x_max, y_max), vari, paksuus)
            luokan_nimi = self.luokat.get(este["luokka"], "Kohde")
            varmuus_prosentti = int(este["varmuus"] * 100)
            
            info_teksti = f"{luokan_nimi} {varmuus_prosentti}% [{teksti}]"
            cv2.putText(naytettava_kuva, info_teksti, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, vari, 2)

            # --- PÄIVITETÄÄN TUTKA ---
            # Skaalataan laatikon koordinaatit tutkan pieneen kokoon
            suhteellinen_x = x_keski / leveys
            suhteellinen_y = y_keski / korkeus
            
            tutka_kohde_x = int(suhteellinen_x * tutka_koko)
            tutka_kohde_y = int(suhteellinen_y * tutka_koko)
            
            # Piirretään piste tutkaan samalla värillä (Vihreä tai Punainen)
            cv2.circle(tutka, (tutka_kohde_x, tutka_kohde_y), 4, vari, -1)

        # 3. YHDISTETÄÄN TUTKA PÄÄKUVAAN (Picture-in-Picture)
        # Laitetaan tutka ruudun oikeaan yläkulmaan (10 pikselin marginaali)
        marginaali = 10
        y_alku = marginaali
        y_loppu = marginaali + tutka_koko
        x_alku = leveys - tutka_koko - marginaali
        x_loppu = leveys - marginaali
        
        # Upotetaan tutka kuvaan
        naytettava_kuva[y_alku:y_loppu, x_alku:x_loppu] = tutka
        # Piirretään tyylikäs valkoinen reunus tutkan ympärille
        cv2.rectangle(naytettava_kuva, (x_alku, y_alku), (x_loppu, y_loppu), (255, 255, 255), 1)

        # Palautetaan tulos
        return naytettava_kuva