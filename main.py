import json
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.data import MetadataCatalog, DatasetCatalog
from tqdm import tqdm

from .transform import fit_transform
from  import reference_detection as rd
import  as at
import .utils.visualize


# def main():

#     data_dir = r"./notebooks/data/table"
#     json_fp = os.path.join(data_dir, "anno.json")
#     arr_fp = os.path.join(data_dir, "anno.npz")
#     with open(json_fp, "r") as fp:
#         mappings = json.load(fp)

#     with np.load(arr_fp) as arrs:
#         anno_dict = {img: {"heads": arrs[f"{prefix}heads"],
#                            "feet": arrs[f"{prefix}feet"]}
#                      for img, prefix in mappings.items()}

#     annotations = anno_dict["D:\\University\\2020-2021\\Internship\\\\notebooks\\data\\table\\img_03.jpg"]
#     # feet and heads have been swapped in annotations
#     reference = annotations["feet"], annotations["heads"]
#     height = 0.095  # m
#     STD = 0.01  # m
#     img = Image.open(
#         "D:\\University\\2020-2021\\Internship\\\\notebooks\\data\\table\\img_03.jpg")

#     image_coords = np.array(
#         [[1216, 1398], [2215, 1754], [3268, 1530], [2067, 1282]])
#     np.random.seed(0)
#     return fit_transform(img, reference, height, STD, image_coords, iters=1e4)


def main():
    # Local to my machine, not on Github!
    p = r"D:..\jonasdata\fluid_res_310\images"
    p = r"D:..\dataset2\images"
    renders = os.listdir(p)
    renders[:5]  # Show first 5
    # Used twice
    paths = list(
        map(lambda x: p + "\\" + x, renders))
    imgs = map(Image.open, paths)
    arrs = map(lambda x: np.asarray(x)[..., :3], imgs)
    predictor, cfg = rd.load_model(return_cfg=True)

    preds = map(predictor, arrs)

    instance_dicts = map(lambda x: rd.instances_to_dict(x, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes")), preds)

    references = map(rd.extract_reference, instance_dicts)

    new_images = map(lambda x: at.utils.visualize.visualize(x[0], x[1], False), zip(paths, references))
    # Save images
    for render, img in tqdm(zip(renders, new_images)):
        path = "notebooks\\data\\heads_feet\\" + render
        # path = "notebooks\\data\\no_tolerance\\" + render
        img.save(path)


if __name__ == "__main__":
    # points = [main() for i in range(10)]
    # print(points)
    # print(list(map(lambda x: (x == points[0]).all(), points)))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(*points[0][:, :2].T, label="Table corners")
    # # ax.scatter(*notebook_real[:,:2].T, label="Notebook corners")
    # ax.legend()
    # plt.show()
    main()
