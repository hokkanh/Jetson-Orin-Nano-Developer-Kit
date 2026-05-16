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


    print("\n=== VAIHE 2: SYVYYSKAMERA - TAKTINEN VAARA-ALUE ===")
    lukija_syvyys = MCAPReader(data_dir="data/mcap_files", kohdetiedosto=tiedosto, topic="/camera/camera/depth/image_rect_raw")
    
    for syvyyskuva in lukija_syvyys.lue_kuvat_generaattorina():
        
        # --- 1. SOKEIDEN PISTEIDEN KORJAUS ---
        korjattu_syvyys = syvyyskuva.copy()
        korjattu_syvyys[korjattu_syvyys == 0] = 10000 # Pakotetaan sokeat pisteet 10 metrin päähän
        
        # --- 2. VAARA-ALUEEN LEIKKAUS (Laajennettu tutka) ---
        minimietaisyys = 500   # 0.5m (Estää dronen omien osien näkymisen)
        maksimietaisyys = 8000 # 8m (Nyt puut näkyvät paljon aiemmin!)
        
        vaara_maski = cv2.inRange(korjattu_syvyys, minimietaisyys, maksimietaisyys)

        # --- 3. UUSI ASE: HORISONTTIFILTTERI (Maan pinnan tuhoaja) ---
        kuvan_korkeus = vaara_maski.shape[0]
        # Määritetään, mistä maa alkaa. 0.65 tarkoittaa, että ylin 65% on ilmatilaa, alin 35% on maata.
        maan_raja = int(kuvan_korkeus * 0.65) 
        
        # Pakotetaan kaikki maan_rajan alapuolella olevat pikselit mustaksi (0)
        vaara_maski[maan_raja:kuvan_korkeus, :] = 0

        # --- 4. ROSKIEN SUODATUS ---
        kernel = np.ones((7, 7), np.uint8)
        puhdas_maski = cv2.morphologyEx(vaara_maski, cv2.MORPH_OPEN, kernel)

        # --- 5. MUODON TUNNISTUS JA ESTEIDEN PIIRTO ---
        contours, _ = cv2.findContours(puhdas_maski, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        naytettava_kuva = cv2.cvtColor(puhdas_maski, cv2.COLOR_GRAY2BGR)
        
        # Piirretään myös hieno visuaalinen viiva ruudulle näyttämään, missä horisonttifiltteri menee
        cv2.line(naytettava_kuva, (0, maan_raja), (naytettava_kuva.shape[1], maan_raja), (255, 0, 0), 1)
        cv2.putText(naytettava_kuva, "MAA/TUTKA OFF", (10, kuvan_korkeus - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        for cnt in contours:
            pinta_ala = cv2.contourArea(cnt)
            
            if pinta_ala < 800:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(naytettava_kuva, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(naytettava_kuva, "ESTE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Syvyyskamera (Taktinen Vaara-alue)", naytettava_kuva)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Vaihe 2 keskeytetty manuaalisesti.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()