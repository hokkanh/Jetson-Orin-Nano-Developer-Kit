from dataReading import MCAPReader
import cv2
import numpy as np
from flask import Flask, Response

# Alustetaan kevyt Web-palvelin
app = Flask(__name__)

def generoi_videovirta():
    # HUOM: Varmista, että "flyover.mcap" on oikea tiedostonimi Jetsonilla!
    tiedosto = "dualtarget.mcap" 
    print("=== SYVYYSKAMERA JA KIRURGINEN PAIKKAUS (STRIIMAUS) ===")
    
    lukija_syvyys = MCAPReader(
        data_dir="data/mcap_files", 
        kohdetiedosto=tiedosto, 
        topic="/camera/camera/depth/image_rect_raw"
    )
    
    for syvyyskuva in lukija_syvyys.lue_kuvat_generaattorina():
        # 1. Skaalataan data 8-bittiseksi
        skaalattu = np.clip(syvyyskuva * (255.0 / 8000.0), 0, 255).astype(np.uint8)
        
        # VAIHE A: Sokeat pisteet ja nollakohdat
        _, pohjamaski = cv2.threshold(skaalattu, 5, 255, cv2.THRESH_BINARY_INV)
        
        # VAIHE B: Maskin laajennus
        ydin = np.ones((5, 5), np.uint8)
        maski = cv2.dilate(pohjamaski, ydin, iterations=1)
        
        # VAIHE C: Tausta-arvaus
        arvattu_tausta = cv2.medianBlur(skaalattu, 21)
        
        # VAIHE D: Kirurginen paikkaus
        paikattu_kuva = skaalattu.copy()
        paikattu_kuva[maski == 255] = arvattu_tausta[maski == 255]
        
        # 2. Värjäys (JET-lämpökartta)
        varitetty_syvyys = cv2.applyColorMap(paikattu_kuva, cv2.COLORMAP_JET)

        # --- WEB-STRIIMAUS TAIKA (Korvaa cv2.imshow:n) ---
        # Pakataan prosessoitu kuva lennosta JPEG-muotoon
        ret, buffer = cv2.imencode('.jpg', varitetty_syvyys)
        if not ret:
            continue
            
        frame = buffer.tobytes()
        # Lähetetään videofreimi HTTP-vastauksena selaimeen
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Reitti, joka tarjoilee videon, kun avaat selaimen
@app.route('/')
def video_feed():
    return Response(generoi_videovirta(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚀ALGORITMI VALMIINA!🚀")
    print("Mene oman läppärin selaimella osoitteeseen: http://<jetsonin_ip_osoite>:5000")
    # host='0.0.0.0' sallii yhteydet kaikista samassa Wi-Fi:ssä olevista koneista
    app.run(host='0.0.0.0', port=5000, debug=False)