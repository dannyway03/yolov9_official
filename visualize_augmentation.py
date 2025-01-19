"""
Script to visualize the images that will be actually fed to the YOLO model. This simple script
helps to understand and debug what the model actually sees.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml

from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_img_size, colorstr, LOGGER, init_seeds

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def draw_bounding_boxes(img, targets, ax):
    """Draw bounding boxes on the image."""
    h, w = img.shape[:2]
    for box in targets:
        x_center, y_center, width, height = box[2] * w, box[3] * h, box[4] * w, box[5] * h
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)


def visualize_augmentation(train_loader, rows: int = 2, cols: int = 2, pages: str = 'all') -> None:
    total_images = len(train_loader.dataset)
    images_per_page = rows * cols
    num_pages = (total_images + images_per_page - 1) // images_per_page

    if pages == 'all':
        pages = range(num_pages)
    else:
        pages = [int(pages)]

    for page in pages:
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        fig.suptitle(f'Augmented Images - Page {page + 1}', fontsize=16)
        axs = axs.flatten()

        for i, (img, targets, _, _) in enumerate(train_loader):
            if i < page * images_per_page:
                continue
            if i >= (page + 1) * images_per_page:
                break

            img = img[0].permute(1, 2, 0).cpu().numpy()  # Assuming batch size of 1
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]

            axs[i % images_per_page].imshow(img)
            draw_bounding_boxes(img, targets, axs[i % images_per_page])
            axs[i % images_per_page].axis('off')

        plt.show()


def main(opt: argparse.Namespace) -> None:
    # Set seed
    init_seeds(opt.seed, deterministic=True)

    # Check dataset
    data_dict = check_dataset(opt.data)
    train_path, val_path = data_dict['train'], data_dict['val']
    imgsz = check_img_size(opt.imgsz, opt.gs)

    # Hyperparameters
    hyp = str(opt.hyp)
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    hyp['anchor_t'] = 5.0
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Create dataloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              1,
                                              opt.gs,
                                              False,
                                              hyp=hyp,
                                              augment=True,
                                              cache=False,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=opt.workers,
                                              image_weights=False,
                                              close_mosaic=True,
                                              quad=False,
                                              prefix=colorstr('Test Augmentation: '),
                                              shuffle=True,
                                              min_items=0)

    visualize_augmentation(train_loader, opt.rows, opt.cols, opt.pages)


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml',
                        help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml',
                        help='hyperparameters path')
    parser.add_argument('--imgsz', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--workers', type=int, default=8,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--gs', type=int, default=32, help='grid size (max stride)')
    parser.add_argument('--rows', type=int, default=2, help='Number of rows of images to display')
    parser.add_argument('--cols', type=int, default=2,
                        help='Number of columns of images to display')
    parser.add_argument('--pages', type=str, default='all',
                        help='Pages to display (all or specific page number)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed to make augmentations reproducible')

    return parser.parse_args()


if __name__ == "__main__":
    # Example CLI usage:
    # python visualize_augmentation.py \
    #   --data plate_detection_dataset/dataset.yaml \
    #   --imgsz 640 \
    #   --hyp 'data/hyps/custom-hyperparameters.yaml'
    opt = parse_opt()
    main(opt)
