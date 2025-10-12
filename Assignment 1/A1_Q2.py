import cv2
import numpy as np

def invisible_cloak():
    cap = cv2.VideoCapture(0)
    background = None
    captured = False

    print("➡ Keep camera still and press ENTER to capture background.")
    print("➡ After that, green cloth will turn invisible.")
    print("➡ Press 'q' to quit anytime.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural view
        frame = cv2.flip(frame, 1)

        # If background not captured yet
        if not captured:
            display_text = "Press ENTER to Capture Background"
            bg_preview = frame.copy()
        else:
            display_text = "Invisible Cloak Active"
            bg_preview = background.copy()

        # --- Text labels ---
        cv2.putText(bg_preview, "BACKGROUND", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "LIVE VIEW", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # --- Once background is captured, apply cloak effect ---
        if captured:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 🎨 Refined green color ranges
            lower_green = np.array([35, 60, 40])     # ignore dull light-green reflections
            upper_green = np.array([85, 255, 255])
            mask1 = cv2.inRange(hsv, lower_green, upper_green)

            # detect slightly darker greens (shadowed cloth)
            lower_dark_green = np.array([25, 40, 30])
            upper_dark_green = np.array([95, 255, 255])
            mask2 = cv2.inRange(hsv, lower_dark_green, upper_dark_green)

            # Combine both masks
            mask = cv2.bitwise_or(mask1, mask2)

            # 🧹 Morphological operations to clean up mask
            # Opening removes noise, Dilation enlarges the mask slightly to cover edges
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

            # Smooth the mask edges to prevent sharp artifacts
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # Invert mask to get everything except green
            inv_mask = cv2.bitwise_not(mask)

            # Replace green with captured background
            cloak_area = cv2.bitwise_and(background, background, mask=mask)
            visible_area = cv2.bitwise_and(frame, frame, mask=inv_mask)
            final = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)
        else:
            final = frame.copy()

        # Combine left (background) and right (invisible)
        combined = np.hstack((bg_preview, final))
        cv2.putText(combined, display_text, (20, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Left: Background | Right: Invisible Cloak", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER key
            background = frame.copy()
            captured = True
            print("✅ Background captured and locked!")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    invisible_cloak()
