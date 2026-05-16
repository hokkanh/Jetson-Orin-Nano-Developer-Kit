from dataReading import MCAPReader
from detection import YOLODetector
from visualization import Visualizer
import cv2
import numpy as np
from flask import Flask, Response

# Alustetaan kevyt Web-palvelin
app = Flask(__name__)

def generoi_videovirta():
    tiedosto = "flyover.mcap" 
    
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
            
        # Nyt silmat.piirra_hud palauttaa valmiin kuvan!
        naytettava_kuva = silmat.piirra_hud(raakakuva, viimeisimmat_esteet)
        
        # --- WEB-STRIIMAUS TAIKA (Pakataan kuva JPEG:ksi livenä) ---
        _, buffer = cv2.imencode('.jpg', naytettava_kuva)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    print("\n=== VAIHE 2: SYVYYSKAMERA JA KIRURGINEN PAIKKAUS ===")
    lukija_syvyys = MCAPReader(data_dir="data/mcap_files", kohdetiedosto=tiedosto, topic="/camera/camera/depth/image_rect_raw")
    
    for syvyyskuva in lukija_syvyys.lue_kuvat_generaattorina():
        skaalattu = np.clip(syvyyskuva * (255.0 / 8000.0), 0, 255).astype(np.uint8)
        _, pohjamaski = cv2.threshold(skaalattu, 5, 255, cv2.THRESH_BINARY_INV)
        ydin = np.ones((5, 5), np.uint8)
        maski = cv2.dilate(pohjamaski, ydin, iterations=1)
        arvattu_tausta = cv2.medianBlur(skaalattu, 21)
        paikattu_kuva = skaalattu.copy()
        paikattu_kuva[maski == 255] = arvattu_tausta[maski == 255]
        varitetty_syvyys = cv2.applyColorMap(paikattu_kuva, cv2.COLORMAP_JET)

        # --- WEB-STRIIMAUS TAIKA ---
        _, buffer = cv2.imencode('.jpg', varitetty_syvyys)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Tämä reitti tarjoilee videon selaimeen
@app.route('/')
def video_feed():
    return Response(generoi_videovirta(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚀 ALGORITMI VALMIINA! Mene läppärin selaimella osoitteeseen: http://<jetsonin_ip_osoite>:5000")
    # host='0.0.0.0' tarkoittaa, että Jetson sallii yhteydet kaikista Wi-Fi-verkon laitteista
    app.run(host='0.0.0.0', port=5000, debug=False)