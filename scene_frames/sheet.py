
import json,sys
from PIL import Image,ImageDraw,ImageFont
S=json.load(open('scenes.json')); L=json.load(open('labels.json')) if len(sys.argv)>1 else {}
W=480; cols=3
ims=[Image.open(s['file']) for s in S]
h=int(W*ims[0].height/ims[0].width); cap=150 if L else 60
rows=(len(S)+cols-1)//cols
sheet=Image.new('RGB',(W*cols,(h+cap)*rows),'white'); d=ImageDraw.Draw(sheet)
try: f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',26); f2=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
except: f=f2=ImageFont.load_default()
import textwrap
for k,(s,im) in enumerate(zip(S,ims)):
    x=(k%cols)*W; y=(k//cols)*(h+cap)
    sheet.paste(im.resize((W,h)),(x,y))
    d.text((x+8,y+h+4),f"{s['i']}  {s['start']:.1f}-{s['end']:.1f}s",fill='black',font=f)
    lab=L.get(str(s['i']),'')
    for j,line in enumerate(textwrap.wrap(lab,38)[:4]): d.text((x+8,y+h+36+j*25),line,fill='#333',font=f2)
sheet.save('sheet.jpg' if L else 'raw.jpg',quality=90)
