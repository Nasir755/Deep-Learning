# Automated Document Segmentation & Recursive Slip Splitting

This repository contains a robust Computer Vision pipeline designed to detect, segment, and crop individual payment slips from a composite image. Developed as part of a Computer Vision quiz, this tool handles complex scenarios including variable lighting, scanner margins, and vertically stacked receipts.

## 🚀 Key Features

* **Adaptive Preprocessing:** Automatically removes scanner frame margins (Top: 22px, Bottom: 33px, Right: 16px, Left: 26px) to eliminate boundary noise.
* **Morphological Continuity:** Employs a tall vertical kernel ($120 \times 1$) and Canny Edge Detection to bridge text gaps within a single receipt, ensuring unified segmentation.
* **Recursive Sobel Splitting:** For "Mega-Slips" (stacked receipts exceeding 1100px in height), the model applies a **Horizontal Sobel Operator** to generate a gradient projection profile.
* **Multi-Slip Decomposition:** The algorithm recursively identifies "valleys" (regions of low horizontal intensity) to cleanly separate stacked receipts.

## 📸 Evaluation & Results

The system generates an `annotated_evaluation.jpg` file to provide visual proof of the detection logic.

### **Detection Map**
![Annotated Evaluation](/Assets/annotated_evaluation.jpg)

* **Green Boxes:** Successfully detected standard slips.
* **Yellow Boxes:** Slips identified and separated via the Recursive Sobel Splitting module.

## 🛠️ Installation & Usage

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/Muntazir-43/Computer-Vision-Quiz-01.git](https://github.com/Muntazir-43/Computer-Vision-Quiz-01.git)
    cd Computer-Vision-Quiz-01
    ```

2.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Pipeline**:
    Ensure your target image is named `input_image.jpeg` in the root directory, then run:
    ```bash
    python CV_QUIZ_1.py
    ```

## 📝 Technical Implementation Details

| Module | Technique | Purpose |
| :--- | :--- | :--- |
| **Preprocessing** | Coordinate Cropping | Scanner border removal |
| **Segmentation** | Morphological Closing | Bridging horizontal text gaps |
| **Analysis** | Horizontal Sobel | Finding cutting points in tall blocks |
| **Recursion** | Depth-First Search | Splitting $N$ number of stacked slips |

---

## 👤 Author
**Asghar Ali**

**Roll No:** 2023-SE-42  
