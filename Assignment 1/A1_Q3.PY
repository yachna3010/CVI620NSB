# A1_Q3.py — Final Version (Meets All Requirements)
# -------------------------------------------------
# Author: <Your Name>
# Course: CVI620
# Description:
#   Modular interactive photo editor using OpenCV & NumPy.
#   Supports multiple transformations in a row, undo, and history.
#   Uses Matplotlib for side-by-side [Original | Preview] display.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import copy


# ---------- Utility Function ----------
def show_images(before, after, title="Preview"):
    """Display original and processed images side-by-side until closed."""
    plt.close("all")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show(block=True)  # wait for user to close window


# ---------- Operations ----------
def adjust_brightness(img):
    value = int(input("Enter brightness adjustment (-100 to 100): "))
    result = cv2.convertScaleAbs(img, alpha=1, beta=value)
    show_images(img, result, f"Brightness {value:+d}")
    return result, f"Brightness {value:+d}"


def adjust_contrast(img):
    alpha = float(input("Enter contrast factor (0.5–3.0): "))
    result = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    show_images(img, result, f"Contrast x{alpha:.2f}")
    return result, f"Contrast x{alpha:.2f}"


def convert_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    show_images(img, gray_bgr, "Grayscale")
    return gray_bgr, "Converted to Grayscale"


def add_padding(img):
    print("\nBorder types:\n1. constant\n2. reflect\n3. replicate\n4. wrap")
    border_types = {
        1: cv2.BORDER_CONSTANT,
        2: cv2.BORDER_REFLECT,
        3: cv2.BORDER_REPLICATE,
        4: cv2.BORDER_WRAP
    }

    b_choice = int(input("Choose border type (1-4): "))
    pad = int(input("Enter padding size (in pixels): "))
    color = (0, 0, 0)
    if b_choice == 1:
        color = tuple(map(int, input("Enter border color (B G R): ").split()))

    result = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                border_types[b_choice], value=color)
    show_images(img, result, "Padded Image")
    return result, f"Padded {pad}px, Type={b_choice}"


def apply_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Threshold Options:\n1. Binary\n2. Binary Inverse")
    t_choice = int(input("Choose threshold type (1-2): "))
    thresh_type = cv2.THRESH_BINARY if t_choice == 1 else cv2.THRESH_BINARY_INV
    _, result = cv2.threshold(gray, 127, 255, thresh_type)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    show_images(img, result_bgr, "Thresholded")
    return result_bgr, f"Threshold ({'Binary' if t_choice == 1 else 'Inverse'})"


def blend_images(img):
    # Default 2nd image path
    path = r"C:\Users\yachn\Documents\cvi\img.jpg"
    if not os.path.exists(path):
        path = input("Enter path to second image: ")

    if not os.path.exists(path):
        print("❌ Second image not found.")
        return img, "Blend failed (missing second image)."

    alpha = float(input("Enter alpha (0–1): "))
    img2 = cv2.imread(path)
    img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    blended = cv2.addWeighted(img, alpha, img2, 1 - alpha, 0)
    show_images(img, blended, f"Blended α={alpha:.2f}")
    return blended, f"Blended with {os.path.basename(path)} α={alpha:.2f}"


# ---------- Main Program ----------
def main():
    path = "new.jpg"  # automatically load your main image

    if not os.path.exists(path):
        print("❌ File not found. Make sure 'new.jpg' is in the same folder.")
        return

    img = cv2.imread(path)
    history = [copy.deepcopy(img)]
    actions = []

    while True:
        print("\n==== Mini Photo Editor ====")
        print("1. Adjust Brightness")
        print("2. Adjust Contrast")
        print("3. Convert to Grayscale")
        print("4. Add Padding")
        print("5. Apply Thresholding")
        print("6. Blend with Another Image")
        print("7. Undo Last Operation")
        print("8. View History of Operations")
        print("9. Exit")

        choice = input("Enter choice (1–9): ")

        if choice == "1":
            img, action = adjust_brightness(img)
        elif choice == "2":
            img, action = adjust_contrast(img)
        elif choice == "3":
            img, action = convert_grayscale(img)
        elif choice == "4":
            img, action = add_padding(img)
        elif choice == "5":
            img, action = apply_threshold(img)
        elif choice == "6":
            img, action = blend_images(img)
        elif choice == "7":
            if len(history) > 1:
                history.pop()
                img = copy.deepcopy(history[-1])
                print("✅ Undo successful.")
                continue
            else:
                print("⚠️ No more undo steps.")
                continue
        elif choice == "8":
            print("\n--- Operation History ---")
            if not actions:
                print("(No actions yet)")
            for i, a in enumerate(actions, 1):
                print(f"{i}. {a}")
            continue
        elif choice == "9":
            save = input("Do you want to save the final image? (y/n): ").strip().lower()
            if save == "y":
                filename = input("Enter filename to save (e.g., output.jpg): ")
                cv2.imwrite(filename, img)
                print(f"✅ Image saved as {filename}")
            print("\nSession Summary:")
            for a in actions:
                print("•", a)
            print("👋 Exiting Photo Editor. Goodbye!")
            break
        else:
            print("Invalid option. Try again.")
            continue

        # Record state for undo and history
        history.append(copy.deepcopy(img))
        actions.append(action)
        print("\n✔ Operation complete! Close the image window to continue...")


if __name__ == "__main__":
    main()
