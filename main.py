# Kuvitteellinen main.py arkkitehtuurin havainnollistamiseksi
from dataReading import MCAPReader
from detection import YOLODetector

lukija = MCAPReader(data_dir="data/mcap_files", kohdetiedosto="dualtarget.mcap")
aivot = YOLODetector()

for kuva in lukija.lue_kuvat_generaattorina():
    esteet = aivot.etsi_esteet(kuva)
    # Tämän jälkeen listassa 'esteet' on siististi kaikki ruudulta löytyneet laatikot!