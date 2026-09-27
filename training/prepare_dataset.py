"""Split a class-per-folder dataset into train/val/test and optionally augment the train split.

    python prepare_dataset.py --src ../data/raw --dst ../data/split --augment 5

Augmentation is applied to the training split only so validation/test numbers stay honest.
"""
import argparse
import os
import random
import shutil

import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def split(src, dst, train_frac, val_frac, seed):
    rng = random.Random(seed)
    if os.path.exists(dst):
        shutil.rmtree(dst)

    for cls in sorted(os.listdir(src)):
        cls_dir = os.path.join(src, cls)
        if not os.path.isdir(cls_dir):
            continue
        images = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS))
        if not images:
            print(f"{cls}: no images, skipped")
            continue
        rng.shuffle(images)

        n_train = int(train_frac * len(images))
        n_val = int(val_frac * len(images))
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for name, files in splits.items():
            out_dir = os.path.join(dst, name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(cls_dir, f), os.path.join(out_dir, f))
        print(f"{cls}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


def augment_train(train_dir, per_image, img_size):
    # Imported lazily so the split step works without TensorFlow installed.
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=10,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
    )
    for cls in sorted(os.listdir(train_dir)):
        cls_dir = os.path.join(train_dir, cls)
        originals = [f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS)]
        print(f"[INFO] Augmenting {cls}: {len(originals)} images x {per_image}")
        for f in originals:
            x = img_to_array(load_img(os.path.join(cls_dir, f), target_size=(img_size, img_size)))
            x = np.expand_dims(x, 0)
            flow = datagen.flow(x, batch_size=1, save_to_dir=cls_dir,
                                save_prefix=f"aug_{os.path.splitext(f)[0]}", save_format="jpg")
            for _ in range(per_image):
                next(flow)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="../data/raw")
    parser.add_argument("--dst", default="../data/split")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--augment", type=int, default=0, help="Augmented copies per training image (0 = off)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split(args.src, args.dst, args.train, args.val, args.seed)
    if args.augment > 0:
        augment_train(os.path.join(args.dst, "train"), args.augment, args.img_size)
    print(f"[INFO] Dataset ready in {args.dst}")


if __name__ == "__main__":
    main()
