# video2scenario (YOLOv8 + homography → trajectories → OpenSCENARIO template)

Pipeline minima per convertire un video (dashcam/roadside) in traiettorie in metri e (opzionale) esportare un file OpenSCENARIO.

## Requisiti
- Python 3.10+
- GPU NVIDIA opzionale (migliora le prestazioni), CUDA se disponibile
- `pip install -r requirements.txt`

## Flusso
1. **Seleziona 4 punti** sul piano stradale che formino un rettangolo reale noto (es. 3.5 m x 10 m).
2. **Tracking** con YOLOv8 + ByteTrack → CSV di bounding box e ID.
3. **Traiettorie in metri** applicando l’omografia ai punti piede del bbox → JSON con traiettorie lisciate.
4. **(Opzionale) Export OpenSCENARIO** (template) da perfezionare in base alla mappa/SVL.

## Esempio rapido
```bash
# 0) Creare ed attivare un venv/conda a piacere
pip install -r requirements.txt

# 1) Clicca 4 punti (in ordine orario) sul primo frame
python scripts/01_select_homography_points.py \
  --video path/al/video.mp4 \
  --rect_w 3.5 --rect_h 10.0 \
  --out data/homography.npz

# 2) Tracking YOLOv8 (classi default: person, bicycle, car, motorcycle, bus, truck)
python scripts/02_track_yolov8.py \
  --video path/al/video.mp4 \
  --out outputs/tracks.csv \
  --model yolov8n.pt --imgsz 960 --conf 0.25

# 3) Costruisci traiettorie in metri (+ smoothing + velocità)
python scripts/03_build_trajectories.py \
  --video path/al/video.mp4 \
  --tracks outputs/tracks.csv \
  --homography data/homography.npz \
  --out outputs/trajectories.json

# 4) (Opzionale) Esporta template OpenSCENARIO
python scripts/04_export_openx.py \
  --trajectories outputs/trajectories.json \
  --out outputs/scenario.xosc \
  --title "video2scenario demo"
```

### Note importanti
- L’omografia assume **piano stradale** piatto. Oggetti su rampe o ponti possono avere errori.
- Le coordinate risultanti (X, Y) sono in metri nel sistema del rettangolo definito (origine in un angolo).
- Per SVL/OpenSCENARIO, serve spesso riallineare coordinate e mappa: usa il template come base e adatta `RoadNetwork`/routing secondo il tuo setup.

