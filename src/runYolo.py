from ultralytics import YOLO
import os

# method to finf BB's for bird in cub/nab
#  runs yolo model in stream mode
# returns array of (cid, cname, bbox)
# only returns first detected
# no conf thresh
# no check on detected class
# yolo produces center,width,height, db wants upper corner width height
# if no detection return entire size
def runYoloOnImages(db,mod):
# Load a model
  model = YOLO(mod)  # pretrained YOLO11n model
  rlist = []
  err = 0
  count = 0
  for id in db.imdata:
    
    imid = id.Id
    print(imid)
    fname = db.imageFileFromImid(imid)
    print(fname)
    # Run batched inference on a list of images
    results = model(fname, stream=False)  # return a generator of Results objects
    count += 1
    print(count)
    # Process results generator

    for result in results:
      print(result)
      boxes = result.boxes  # Boxes object for bounding box outputs
      print(len(boxes))
      if len(boxes) == 0: # no detection
    #   os.remove(result.path)
        rlist.append((imid, None,None,0,[0,0,result.orig_shape[1],result.orig_shape[0]]))
        err += 1
        continue
      box = boxes[0]
      clname = result.names[box.cls.item()]
      #if clname != 'bird':
      #  rlist.append((imid, None,None,0,[0,0,result.orig_shape[1],result.orig_shape[0]]))
      #  err += 1
      #  continue
      print(f"  {box.cls}, {box.conf}, {result.names[box.cls.item()]}, {box.xywh}")
      cx = int(box.xywh[0][0].item())
      cy = int(box.xywh[0][1].item())
      w = int(box.xywh[0][2].item())
      h = int(box.xywh[0][3].item())
      res = (imid, box.cls.item(), clname,round(box.conf.item(),2),[int(cx-w/2),int(cy-h/2),w,h])
      rlist.append(res)
      
      masks = result.masks  # Masks object for segmentation masks outputs
      keypoints = result.keypoints  # Keypoints object for pose outputs
      probs = result.probs  # Probs object for classification outputs
      obb = result.obb  # Oriented boxes object for OBB outputs
    # f = result.save(filename=f"result_{os.path.basename(result.path)}")  # save to disk
  print(f"{err} not found in {count}")
  return rlist

from birddb import BirdDB
from env import Env
Env.setupEnv()
db = BirdDB.DBFromName('cub_os')
rlist = runYoloOnImages(db,"./yolo/yolo11s_e40.pt")

with open(f'{db.dbdir}/bounding_boxes_yolo_e40.txt','w') as f:
  for l in rlist:
    if len(l) == 2:
      f.write(f"{l[0]} {l[1]}\n")
    else:
      f.write(f'{l[0]} {l[4][0]} {l[4][1]} {l[4][2]} {l[4][3]}\n')
with open(f'{db.dbdir}/res_yolo_e40.csv','w') as f:
  for l in rlist:
    if len(l) == 2:
      f.write(f"{l[0]},{l[1]}\n")
    else:
      f.write(f'{l[0]},{l[1]},{l[2]},{l[3]},{l[4][0]},{l[4][1]},{l[4][2]},{l[4][3]}\n')
      
   

  