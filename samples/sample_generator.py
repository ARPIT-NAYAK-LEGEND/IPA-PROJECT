import os
from typing import Tuple
import cv2
import numpy as np


def generate_object_pair(width: int = 640, height: int = 480) -> Tuple[np.ndarray, np.ndarray]:
    base = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        v = int(120 + 20 * np.sin(y / 15.0) + (y % 4) * 2)
        base[y, :, :] = [max(0, v - 40), max(0, v - 20), v]

    cv2.rectangle(base, (60, 80), (320, 260), (45, 45, 48), -1)
    cv2.rectangle(base, (75, 95), (305, 245), (190, 210, 225), -1)
    cv2.putText(base, "RESEARCH NOTEBOOK", (95, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 50, 60), 2)

    cv2.circle(base, (480, 160), 38, (240, 240, 245), -1)
    cv2.circle(base, (480, 160), 30, (50, 30, 20), -1)

    img_a = base.copy()
    img_b = base.copy()

    cv2.line(img_a, (360, 220), (440, 280), (20, 30, 180), 8)
    cv2.line(img_a, (435, 275), (445, 285), (200, 200, 200), 4)

    cv2.rectangle(img_b, (380, 240), (460, 380), (25, 25, 25), -1)
    cv2.rectangle(img_b, (386, 250), (454, 370), (0, 140, 220), -1)
    cv2.circle(img_b, (540, 200), 20, (60, 140, 200), -1)

    return img_a, img_b


def generate_document_pair(width: int = 640, height: int = 480) -> Tuple[np.ndarray, np.ndarray]:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)

    cv2.rectangle(canvas, (40, 30), (width - 40, 90), (35, 45, 60), -1)
    cv2.putText(canvas, "GLOBAL TECH AUDIT - INVOICE #8942", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(canvas, "Client: Acme Corporation", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.putText(canvas, "Date: November 02, 2026", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.putText(canvas, "Status: Approved", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 120, 40), 2)

    cv2.line(canvas, (45, 220), (width - 45, 220), (180, 180, 180), 2)
    cv2.putText(canvas, "Item", (55, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 2)
    cv2.putText(canvas, "Hours", (280, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 2)
    cv2.putText(canvas, "Amount", (460, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 2)
    cv2.line(canvas, (45, 260), (width - 45, 260), (180, 180, 180), 1)
    cv2.putText(canvas, "Computer Vision Pipeline", (55, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

    img_a = canvas.copy()
    img_b = canvas.copy()

    cv2.putText(img_a, "40 hrs", (280, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.putText(img_a, "$4,800.00", (460, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 2)
    cv2.putText(img_a, "Authorized Signatory: Dr. Alex Vance", (55, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

    cv2.putText(img_b, "95 hrs", (280, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 20, 20), 1)
    cv2.putText(img_b, "$11,400.00", (460, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 20, 20), 2)
    cv2.putText(img_b, "Authorized Signatory: REJECTED", (55, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 20, 20), 1)
    cv2.rectangle(img_b, (420, 360), (580, 430), (0, 0, 220), 3)
    cv2.putText(img_b, "AUDIT ALERT", (435, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

    return img_a, img_b


def generate_pcb_pair(width: int = 640, height: int = 480) -> Tuple[np.ndarray, np.ndarray]:
    pcb = np.full((height, width, 3), (25, 95, 35), dtype=np.uint8)

    for i in range(8):
        y = 70 + i * 45
        cv2.line(pcb, (60, y), (260, y), (40, 180, 215), 3)
        cv2.line(pcb, (260, y), (340, y + 25), (40, 180, 215), 3)
        cv2.line(pcb, (340, y + 25), (560, y + 25), (40, 180, 215), 3)

    cv2.rectangle(pcb, (230, 160), (390, 320), (25, 25, 25), -1)
    cv2.putText(pcb, "ARM CORTEX", (245, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    for py in range(175, 310, 18):
        cv2.line(pcb, (215, py), (230, py), (180, 180, 180), 2)
        cv2.line(pcb, (390, py), (405, py), (180, 180, 180), 2)

    img_a = pcb.copy()
    img_b = pcb.copy()

    cv2.rectangle(img_a, (120, 180), (160, 210), (30, 120, 200), -1)
    cv2.rectangle(img_a, (120, 240), (160, 270), (30, 120, 200), -1)
    cv2.rectangle(img_a, (460, 180), (500, 210), (30, 120, 200), -1)

    cv2.rectangle(img_b, (120, 180), (160, 210), (30, 120, 200), -1)
    cv2.rectangle(img_b, (120, 245), (130, 265), (160, 160, 160), -1)
    cv2.rectangle(img_b, (150, 245), (160, 265), (160, 160, 160), -1)
    cv2.rectangle(img_b, (460, 180), (500, 210), (30, 120, 200), -1)

    cv2.circle(img_b, (300, 115), 16, (20, 20, 20), -1)
    cv2.line(img_b, (290, 105), (312, 125), (190, 190, 190), 3)

    return img_a, img_b


def generate_architecture_pair(width: int = 640, height: int = 480) -> Tuple[np.ndarray, np.ndarray]:
    scene = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        b = int(220 - y * 0.25)
        g = int(180 - y * 0.2)
        r = int(140 - y * 0.15)
        scene[y, :, :] = [np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)]

    scene[380:, :] = (60, 90, 50)
    cv2.rectangle(scene, (120, 120), (500, 380), (160, 150, 140), -1)
    cv2.rectangle(scene, (100, 100), (520, 125), (80, 75, 70), -1)
    cv2.rectangle(scene, (270, 280), (350, 380), (50, 60, 90), -1)

    img_a = scene.copy()
    img_b = scene.copy()

    for wx in (160, 380):
        for wy in (160, 260):
            cv2.rectangle(img_a, (wx, wy), (wx + 80, wy + 70), (220, 210, 160), -1)
            cv2.line(img_a, (wx + 40, wy), (wx + 40, wy + 70), (50, 50, 50), 2)
            cv2.line(img_a, (wx, wy + 35), (wx + 80, wy + 35), (50, 50, 50), 2)

    for wx in (160, 380):
        for wy in (160, 260):
            color = (80, 220, 255) if (wx, wy) == (160, 160) else (220, 210, 160)
            cv2.rectangle(img_b, (wx, wy), (wx + 80, wy + 70), color, -1)
            cv2.line(img_b, (wx + 40, wy), (wx + 40, wy + 70), (50, 50, 50), 2)
            cv2.line(img_b, (wx, wy + 35), (wx + 80, wy + 35), (50, 50, 50), 2)

    cv2.rectangle(img_b, (365, 225), (475, 250), (40, 40, 45), -1)
    for bx in range(370, 475, 12):
        cv2.line(img_b, (bx, 205), (bx, 225), (50, 50, 55), 2)
    cv2.line(img_b, (365, 205), (475, 205), (30, 30, 35), 3)

    return img_a, img_b


def create_all_samples(output_dir: str = "samples") -> None:
    os.makedirs(output_dir, exist_ok=True)
    generators = {
        "desk_objects": generate_object_pair,
        "document_tampering": generate_document_pair,
        "pcb_defect": generate_pcb_pair,
        "building_facade": generate_architecture_pair,
    }

    for name, gen_fn in generators.items():
        img_a, img_b = gen_fn()
        cv2.imwrite(os.path.join(output_dir, f"{name}_before.png"), cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{name}_after.png"), cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    create_all_samples()
