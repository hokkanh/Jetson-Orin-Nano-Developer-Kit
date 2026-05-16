from dataReading import MCAPFusionReader
from detection import YOLODetector
from visualization import Visualizer
import cv2
import numpy as np

def main():
    print("=== KÄYNNISTETÄÄN ULTRA-OPTIMOITU SENSOR FUSION ===")
    tiedosto = "dualtarget.mcap"

    # Yksi lukija hoitaa koko tiedoston sekunneissa
    lukija = MCAPFusionReader(data_dir="data/mcap_files", kohdetiedosto=tiedosto)
    aivot = YOLODetector()
    silmat = Visualizer()

    # Välimuisti (Cache) uusimmille ruuduille
    viimeisin_infra = None
    viimeisin_syvyys = None
    esteet = []
    infra_laskuri = 0
    paivitys_vali = 5  # YOLO ajetaan vain joka 5. ruutu (80% GPU-säästö!)

    for tyyppi, kuva in lukija.lue_synkronoitu_virta():
        if tyyppi == "infra":
            viimeisin_infra = kuva
            infra_laskuri += 1
            
            # Ajetaan raskas tekoäly vain sykleissä
            if infra_laskuri % paivitys_vali == 0:
                esteet = aivot.etsi_esteet(viimeisin_infra)
                
            # Jos meillä on molemmat sensoritiedot valmiina, piirretään HUD lennosta
            if viimeisin_infra is not None and viimeisin_syvyys is not None:
                hud_kuva = cv2.cvtColor(viimeisin_infra, cv2.COLOR_GRAY2BGR)
                korkeus, leveys = hud_kuva.shape[:2]
                
                for este in esteet:
                    x_keski = este["x"]
                    y_keski = este["y"]
                    w = este["w"]
                    h = este["h"]
                    
                    # --- ULTRA-NOPEA ETÄISYYDEN NOUTO (Ei array-allokaatioita) ---
                    # Luetaan suoraan keskipisteestä
                    z = viimeisin_syvyys[y_keski, x_keski]
                    
                    # Jos keskipiste oli sokea (0), katsotaan nopeasti 4 naapuripikseliä ympäriltä
                    if z == 0:
                        for dy, dx in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
                            ny = max(0, min(korkeus - 1, y_keski + dy))
                            nx = max(0, min(leveys - 1, x_keski + dx))
                            nz = viimeisin_syvyys[ny, nx]
                            if nz > 0:
                                z = nz
                                break
                    
                    # Muutetaan metreiksi
                    etaisyys_m = z / 1000.0 if z > 0 else -1.0
                    
                    # --- PIIRTÄMINEN ---
                    x_min = int(x_keski - (w / 2))
                    y_min = int(y_keski - (h / 2))
                    x_max = int(x_keski + (w / 2))
                    y_max = int(y_keski + (h / 2))
                    
                    # Aina siisti, luotettava vihreä laatikko
                    cv2.rectangle(hud_kuva, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Tulostetaan vain kohteen nimi ja tarkka 3D-etäisyys
                    if etaisyys_m > 0:
                        teksti = f"IHMINEN: {etaisyys_m:.2f}m"
                    else:
                        teksti = "IHMINEN: ETaISYYS HUKASSA"
                        
                    cv2.putText(hud_kuva, teksti, (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Näytetään valmis HUD salamannopeasti (Odotus 1ms = maksiminopeus)
                cv2.imshow("3D Ultra-HUD", hud_kuva)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        elif tyyppi == "depth":
            # Päivitetään syvyyskartta välimuistiin taustalla
            viimeisin_syvyys = kuva

    cv2.destroyAllWindows()
    print("Suoritus päättynyt.")

if __name__ == "__main__":
    main()