from dataReading import MCAPReader
from detection import YOLODetector
from visualization import Visualizer
import cv2

def main():
    print("Käynnistetään taktinen HUD-järjestelmä...")
    
    lukija = MCAPReader(data_dir="data/mcap_files", kohdetiedosto="flyover.mcap")
    aivot = YOLODetector()
    silmat = Visualizer()
    
    kuva_laskuri = 0
    paivitys_vali = 5 # Aja tekoäly vain joka 5. ruutu! (Säästää 80% Jetsonin tehosta)
    viimeisimmat_esteet = [] # Muistaa laatikoiden paikat väliin jätetyillä ruuduilla
    
    for raakakuva in lukija.lue_kuvat_generaattorina():
        kuva_laskuri += 1
        
        # OMINAISUUS 1: FRAME SKIPPING (Tehokkuus)
        # Lasketaan tekoäly uudelleen vain N ruudun välein
        if kuva_laskuri % paivitys_vali == 0:
            viimeisimmat_esteet = aivot.etsi_esteet(raakakuva)
        
        # Silmät piirtävät tulokset ruudulle JOKA ruudulla, jotta video ei töki
        nappain = silmat.piirra_hud(raakakuva, viimeisimmat_esteet)
        
        if nappain == ord('q'):
            print("Käyttäjä keskeytti lennon.")
            break

    cv2.destroyAllWindows()
    print(f"Ajo suoritettu. Käsitelty {kuva_laskuri} ruutua.")

if __name__ == "__main__":
    main()