from dataReading import MCAPReader
from detection import YOLODetector
from visualization import Visualizer
import cv2
import numpy as np

def main():
    tiedosto = "flyover.mcap" # Vaihda tätä halutessasi testata muita!

    print("=== VAIHE 1: TAKTINEN HUD (Infrapuna + YOLO) ===")
    lukija_infra = MCAPReader(data_dir="data/mcap_files", kohdetiedosto=tiedosto, topic="/camera/camera/infra1/image_rect_raw")
    aivot = YOLODetector()
    silmat = Visualizer()
    
    kuva_laskuri = 0
    viimeisimmat_esteet = []
    
    for raakakuva in lukija_infra.lue_kuvat_generaattorina():
        kuva_laskuri += 1
        
        if kuva_laskuri % 3 == 0:
            viimeisimmat_esteet = aivot.etsi_esteet(raakakuva)
            
        nappain = silmat.piirra_hud(raakakuva, viimeisimmat_esteet)
        if nappain == ord('q'):
            print("Vaihe 1 keskeytetty manuaalisesti.")
            break

    cv2.destroyAllWindows()


    print("\n=== VAIHE 2: SYVYYSKAMERA JA KIRURGINEN PAIKKAUS ===")
    lukija_syvyys = MCAPReader(data_dir="data/mcap_files", kohdetiedosto=tiedosto, topic="/camera/camera/depth/image_rect_raw")
    
    for syvyyskuva in lukija_syvyys.lue_kuvat_generaattorina():
        
        # 1. Skaalataan data 8-bittiseksi
        skaalattu = np.clip(syvyyskuva * (255.0 / 8000.0), 0, 255).astype(np.uint8)

        # --- SIGNAALINKÄSITTELY 4.0: KIRURGINEN REIKIEN PAIKKAUS ---
        
        # VAIHE A: Etsitään sokeat pisteet (siniset läiskät eli arvot alle 5)
        # Luodaan "maski", jossa reiät ovat valkoisia ja terve data mustaa
        _, maski = cv2.threshold(skaalattu, 5, 255, cv2.THRESH_BINARY_INV)

        # VAIHE B: Luodaan arvaus taustasta
        # Ajetaan massiivinen (9x9) mediaanisuodatin. Se sumentaa kuvan, mutta 
        # tuhoaa siniset reiät täysin jättäen vain punaoranssia väriä.
        arvattu_tausta = cv2.medianBlur(skaalattu, 9)

        # VAIHE C: Kirurginen siirto
        # Otetaan alkuperäinen kuva, mutta TÄYTETÄÄN reiät arvatulla taustalla.
        # Tämä koskee VAIN niitä sinisiä pisteitä, muu kuva pysyy millimetrin tarkkana!
        paikattu_kuva = skaalattu.copy()
        paikattu_kuva[maski == 255] = arvattu_tausta[maski == 255]

        # 3. VÄRJÄYS
        varitetty_syvyys = cv2.applyColorMap(paikattu_kuva, cv2.COLORMAP_JET)

        # Näytetään tulos
        cv2.imshow("Syvyyskamera (Kirurgisesti Suodatettu)", varitetty_syvyys)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Vaihe 2 keskeytetty manuaalisesti.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()