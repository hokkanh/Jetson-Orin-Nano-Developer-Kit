from dataReading import MCAPReader
from detection import YOLODetector
from visualization import Visualizer
import cv2

def main():
    print("Käynnistetään järjestelmä...")
    
    # 1. Alustetaan kaikki 3 moduulia
    lukija = MCAPReader(data_dir="data/mcap_files", kohdetiedosto="dualtarget.mcap")
    aivot = YOLODetector()
    silmat = Visualizer()
    
    kuva_laskuri = 0
    
    # 2. Aloitetaan datan lypsäminen generaattorilla
    for raakakuva in lukija.lue_kuvat_generaattorina():
        kuva_laskuri += 1
        
        # Vaihe A: Aivot etsivät koordinaatit
        esteet = aivot.etsi_esteet(raakakuva)
        
        # Vaihe B: Silmät piirtävät tulokset ruudulle
        nappain = silmat.piirra_hud(raakakuva, esteet)
        
        # Jos käyttäjä painaa 'q', ohjelma sammuu turvallisesti
        if nappain == ord('q'):
            print("Käyttäjä keskeytti lennon.")
            break

    # Lopuksi siivotaan ikkunat pois
    cv2.destroyAllWindows()
    print(f"Ajo suoritettu. Käsitelty {kuva_laskuri} ruutua.")

if __name__ == "__main__":
    main()