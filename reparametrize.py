from pathlib import Path

import torch

from models.yolo import Model

ROOT_DIR = Path().resolve()

device = torch.device("cpu")
cfg = str(ROOT_DIR / "models/detect/gelan-t.yaml")
n_c = 2
model = Model(cfg, ch=3, nc=n_c, anchors=3)
# model = model.half()
model = model.to(device)
_ = model.eval()
# yolo_path = ROOT_DIR / "yolov9-n.pt"
yolo_path = "/Users/anka/Downloads/runs/train/yolov9-t-640/weights/best.pt"
ckpt = torch.load(yolo_path, map_location='cpu')
model.names = ckpt['model'].names
model.nc = n_c

idx = 0
for k, v in model.state_dict().items():
    if "model.{}.".format(idx) in k:
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
    else:
        while True:
            idx += 1
            if "model.{}.".format(idx) in k:
                break
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
_ = model.eval()

m_ckpt = {'model': model,
          'optimizer': None,
          'best_fitness': None,
          'ema': None,
          'updates': None,
          'opt': None,
          'git': None,
          'date': None,
          'epoch': -1}
torch.save(m_ckpt, ROOT_DIR / "yolov9-t-640-plates-v2-converted.pt")
