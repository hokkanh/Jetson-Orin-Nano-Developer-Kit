from dataReading import MCAPReader
from detection import YOLODetector
from visualization import Visualizer
import cv2
import numpy as np

def main():
    tiedosto = "dualtarget.mcap" # Vaihda tätä halutessasi testata muita!

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


    print("\n=== VAIHE 2: SYVYYSKAMERA JA LEHTIEN SUODATUS ===")
    # Nyt luetaan sama tiedosto, mutta depth-kanavalta!
    lukija_syvyys = MCAPReader(data_dir="data/mcap_files", kohdetiedosto=tiedosto, topic="/camera/camera/depth/image_rect_raw")
    
    for syvyyskuva in lukija_syvyys.lue_kuvat_generaattorina():
        
        # 1. Puristetaan 16-bittinen syvyysdata näytettävään 8-bittiseen muotoon
        # Arvo 8000.0 tarkoittaa, että maksimietäisyys on 8 metriä (kaikki sen yli on mustaa)
        skaalattu = np.clip(syvyyskuva * (255.0 / 8000.0), 0, 255).astype(np.uint8)

        # 2. SIGNAALINKÄSITTELY: Pöllyävien lehtien ja reikäisen kohinan tuhoaminen
        # Ase 1: Mediaanisuodatin (Korjaa yksittäiset virhepikselit)
        suodatettu = cv2.medianBlur(skaalattu, 5)
        
        # Ase 2: Morfologinen avaus (Syö pois lehdet, mutta jättää isot esteet)
        kernel = np.ones((5, 5), np.uint8)
        suodatettu = cv2.morphologyEx(suodatettu, cv2.MORPH_OPEN, kernel)

        # 3. VÄRJÄYS: Tehdään syvyyskartasta tyylikäs värikartta (Lämpökamera-tyyli)
        varitetty_syvyys = cv2.applyColorMap(suodatettu, cv2.COLORMAP_JET)

        # Näytetään tulos
        cv2.imshow("Syvyyskamera (Suodatettu)", varitetty_syvyys)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Vaihe 2 keskeytetty manuaalisesti.")
            break

    cv2.destroyAllWindows()
    print("Kaikki operaatiot suoritettu onnistuneesti.")

if __name__ == "__main__":
    main()