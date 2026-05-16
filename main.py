from dataReading import MCAPReader
from detection import YOLODetector
# from visualizer import Visualizer

def main():
    print("Käynnistetään järjestelmä...")
    
    # Alustetaan osat
    lukija = MCAPReader(data_dir="data/mcap_files", kohdetiedosto="dualtarget.mcap")
    aivot = YOLODetector()
    
    kuva_laskuri = 0
    
    # Lypsetään mcap-tiedostoa kuva kerrallaan
    for syvyyskuva in lukija.lue_kuvat_generaattorina():
        kuva_laskuri += 1
        
        # 1. Annetaan kuva Aivoille, saadaan takaisin lista koordinaatteja
        esteet = aivot.etsi_esteet(syvyyskuva)
        
        # Printataan tulokset testin vuoksi konsoliin
        if esteet:
            print(f"Ruutu {kuva_laskuri}: Löytyi {len(esteet)} kohdetta! Esimerkki: {esteet[0]}")
            
        if kuva_laskuri >= 100:
            break

if __name__ == "__main__":
    main()