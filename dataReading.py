from dataReading import MCAPReader
import cv2
import numpy as np
from flask import Flask, Response

# Alustetaan HTTP-palvelin (Flask) reaaliaikaista datansiirtoa varten
app = Flask(__name__)

def generoi_videovirta():
    # Määritetään käsiteltävä MCAP-tallennustiedosto
    tiedosto = "dualtarget.mcap" 
    print("=== SYVYYSKAMERADATAN REAALIAIKAINEN ENNALLISTAMINEN ===")
    
    lukija_syvyys = MCAPReader(kohdetiedosto=tiedosto)
    
    for syvyyskuva in lukija_syvyys.lue_kuvat_generaattorina():
        # 1. Normalisoidaan etäisyysdata ja kvantisoidaan se 8-bittiseen esitysmuotoon
        skaalattu = np.clip(syvyyskuva * (255.0 / 8000.0), 0, 255).astype(np.uint8)
        
        # VAIHE A: Virheellisten havaintojen ja sokeiden pisteiden segmentointi kynnystyksellä
        _, pohjamaski = cv2.threshold(skaalattu, 5, 255, cv2.THRESH_BINARY_INV)
        
        # VAIHE B: Binäärimaskin morfologinen laajennus (Dilation) raja-alueiden kattamiseksi
        ydin = np.ones((5, 5), np.uint8)
        maski = cv2.dilate(pohjamaski, ydin, iterations=1)
        
        # VAIHE C: Spatiaalisen tausta-estimaatin laskenta epälineaarisella mediaanisuodatuksella
        arvattu_tausta = cv2.medianBlur(skaalattu, 21)
        
        # VAIHE D: Maskipohjainen datan imputointi (Image Inpainting) ja ennallistaminen
        paikattu_kuva = skaalattu.copy()
        paikattu_kuva[maski == 255] = arvattu_tausta[maski == 255]
        
        # 2. Pseudo-värjäys spatiaalisen hahmottamisen parantamiseksi (JET-värikartta)
        varitetty_syvyys = cv2.applyColorMap(paikattu_kuva, cv2.COLORMAP_JET)

        # Enkoodataan prosessoitu matriisi JPEG-muotoon ja jaetaan MJPEG-videovirtana
        ret, buffer = cv2.imencode('.jpg', varitetty_syvyys)
        if not ret:
            continue
            
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# HTTP-päätepiste videovirran välittämiseksi asiakasohjelmalle
@app.route('/')
def video_feed():
    return Response(generoi_videovirta(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚀 ALGORITMI VALMIINA!🚀")
    print("Mene oman läppärin selaimella osoitteeseen: http://<jetsonin_ip_osoite>:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)