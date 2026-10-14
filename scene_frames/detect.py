
import cv2, numpy as np, json
from scenedetect import detect, AdaptiveDetector, ContentDetector
V='in.mov'
scenes = detect(V, ContentDetector(threshold=20, min_scene_len=4))
cap=cv2.VideoCapture(V); fps=cap.get(5); n=int(cap.get(7))
if not scenes: scenes=[(None,None)]
out=[]
for i,(s,e) in enumerate(scenes):
    a,b=s.get_frames(),e.get_frames()
    pad=max(1,(b-a)//6)  # skip transition-blend frames at edges
    best=None
    for f in range(a+pad if b-a>4 else a, b-pad if b-a>4 else b):
        cap.set(1,f); ok,fr=cap.read()
        if not ok: continue
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        sharp=cv2.Laplacian(g,cv2.CV_64F).var()
        contrast=g.std(); mean=g.mean()
        expo=1-abs(mean-115)/115  # prefer mid exposure
        score=np.log1p(sharp)*0.5+np.log1p(contrast)*0.35+max(expo,0)*0.15*5
        if best is None or score>best[0]: best=(score,f,fr)
    if best is None: continue
    p=f'scene_{i+1:02d}.jpg'; cv2.imwrite(p,best[2],[cv2.IMWRITE_JPEG_QUALITY,95])
    out.append(dict(i=i+1,start=round(a/fps,2),end=round(b/fps,2),best=round(best[1]/fps,2),file=p))
json.dump(out,open('scenes.json','w'),indent=1); print(json.dumps(out))
