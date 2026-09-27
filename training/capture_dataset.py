"""Capture labelled training images from a webcam.

Press 's' to save the current frame into <out-dir>/<class-name>/, 'q' to quit.

    python capture_dataset.py --class-name CEO --out-dir ../data/raw --camera 0
"""
import argparse
import os

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-name", required=True, help="Folder name for this class (e.g. CEO)")
    parser.add_argument("--out-dir", default="../data/raw")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--size", type=int, default=224, help="Saved image side length in pixels")
    args = parser.parse_args()

    class_dir = os.path.join(args.out_dir, args.class_name)
    os.makedirs(class_dir, exist_ok=True)
    count = len(os.listdir(class_dir))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    print(f"[INFO] Capturing class '{args.class_name}' into {class_dir} ('s' = save, 'q' = quit)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (args.size, args.size))
        cv2.imshow("Capture - 's' to save, 'q' to quit", resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            path = os.path.join(class_dir, f"{count}.jpg")
            cv2.imwrite(path, resized)
            print(f"Saved: {path}")
            count += 1
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
