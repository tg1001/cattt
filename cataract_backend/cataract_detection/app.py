import os
import gradio as gr
import torch
from PIL import Image

from utils.model_loader import load_model
from utils.preprocessing import preprocess
from utils.inference import run_ensemble
from sanity_check_models import run_sanity_check

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model paths
# ----------------------------
MODEL_A_PATH  = "models/modelA.pth"
MODEL_B1_PATH = "models/modelB1.pth"
MODEL_B2_PATH = "models/modelB2.pth"

# ----------------------------
# Calibration thresholds (FINAL)
# ----------------------------
ZONE_1_MAX = 0.08
ZONE_2A_MAX = 0.10
ZONE_2B_MAX = 0.12
CATARACT_OPACITY_THRESHOLD = 0.70

# ----------------------------
# Sanity check
# ----------------------------
run_sanity_check()

# ----------------------------
# Force HF Xet materialization
# ----------------------------
for p in [MODEL_A_PATH, MODEL_B1_PATH, MODEL_B2_PATH]:
    if not os.path.exists(p):
        raise RuntimeError(f"Model file missing: {p}")
    with open(p, "rb") as f:
        f.read(1)

# ----------------------------
# Load models ONCE
# ----------------------------
modelA  = load_model(MODEL_A_PATH, device)
modelB1 = load_model(MODEL_B1_PATH, device)
modelB2 = load_model(MODEL_B2_PATH, device)

# ----------------------------
# Prediction function
# ----------------------------
def predict(image):
    if image is None:
        return "—", "—", "—", {"Cataract": 0.0, "No Cataract": 0.0}

    img = Image.fromarray(image).convert("RGB")
    img_224, img_256 = preprocess(img)

    result = run_ensemble(
        img_224,
        img_256,
        modelA,
        modelB1,
        modelB2,
        device
    )

    probs = result["severity_probs"]  # [No Cataract, Immature, Mature]

    p_nc = float(probs[0])
    p_cat = 1.0 - p_nc

    # ----------------------------
    # Classification + Action
    # ----------------------------
    if p_nc <= ZONE_1_MAX:
        assessment = "Cataract Present"
        note = (
            "Clear imaging evidence of cataract is detected. "
            "Lens opacity patterns are consistent with clinically significant cataract."
        )
        suggested_action = "Ophthalmologic evaluation recommended."

    elif ZONE_1_MAX < p_nc < ZONE_2A_MAX:
        assessment = "Likely Early (Immature) Cataract"
        note = (
            "Imaging patterns suggest early or immature cataract formation. "
            "Structural lens changes are present but not advanced."
        )
        suggested_action = "Routine monitoring or clinical correlation advised."

    elif ZONE_2A_MAX <= p_nc < ZONE_2B_MAX:
        assessment = "Likely Non-Cataract (Early Overlap)"
        note = (
            "No definitive cataract detected. "
            "Subtle lens patterns may overlap with very early cataract features "
            "or normal physiological variation."
        )
        suggested_action = "Monitoring recommended if symptoms develop."

    else:
        assessment = "No Cataract Detected"
        note = (
            "Lens appearance is consistent with a healthy lens. "
            "No imaging evidence of cataract is observed."
        )
        suggested_action = "No immediate follow-up required."

    # ----------------------------
    # Lens Type
    # ----------------------------
    if p_cat >= CATARACT_OPACITY_THRESHOLD:
        lens_type = (
            "Likely natural lens (low confidence)\n"
            "(Lens assessment unreliable due to cataract opacity)"
        )
    else:
        lens_type = result["lens_type"]

    confidence_dist = {
        "Cataract": round(p_cat, 3),
        "No Cataract": round(p_nc, 3)
    }

    return assessment, note, suggested_action, lens_type, confidence_dist

# ----------------------------
# UI
# ----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🩺 Cataract Screening — Research Demo
    **Calibrated deep-learning ensemble optimized for cataract screening.**

    ⚠️ Research & educational use only  
    Not a diagnostic or clinical medical device
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Eye Image",
                type="numpy",
                height=260
            )

        with gr.Column(scale=1):
            assessment_out = gr.Textbox(label="Result", interactive=False)
            note_out = gr.Textbox(label="Explanation", interactive=False)
            action_out = gr.Textbox(label="Suggested Action", interactive=False)
            lens_out = gr.Textbox(label="Lens Type (Context-Aware)", interactive=False)

    cataract_plot = gr.Label(label="Cataract Confidence")
    
    # ----------------------------
    # Confidence explanation (Layer 1)
    # ----------------------------
    gr.Markdown("""
    **How to read this score:**  
    This percentage reflects internal model confidence, not a diagnosis.  
    Final results use conservative screening ranges (e.g., clear non-cataract ≥ 12%) to reduce false positives.
    """)

    # ----------------------------
    # Confidence explanation (Layer 2 – expandable)
    # ----------------------------
    with gr.Accordion("How to Read Cataract Confidence", open=False):
        gr.Markdown("""
        This percentage does **not** mean you have cataract.

        It shows how strongly the model’s internal patterns lean toward
        **cataract vs non-cataract features** in the image.

        The system is **intentionally conservative**.  
        The final result is **not decided by this percentage alone**, but by predefined
        screening ranges designed to avoid over-calling disease.

        **In simple terms, the system interprets the confidence like this:**

        - **Strong cataract signal**  
          → Very low non-cataract probability (typically ≤ 8%)  
          → Reported as **Cataract Present**

        - **Early / borderline signal**  
          → Low but not definitive non-cataract probability  
          → Reported as **Likely Early (Immature) Cataract**

        - **Overlap / uncertainty zone**  
          → Mixed cataract and non-cataract signals  
          → Reported as **Likely Non-Cataract (Early Overlap)**

        - **Clear non-cataract signal**  
          → Non-cataract probability ≥ 12%  
          → Reported as **No Cataract Detected**

        Because of this conservative design, the model may display a **high cataract confidence**
        and still report *early*, *overlap*, or *no cataract* rather than making a definitive call.

        This behavior is intentional and helps reduce false positives in screening.
        """)

    predict_btn = gr.Button("Run Screening")

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            assessment_out,
            note_out,
            action_out,
            lens_out,
            cataract_plot
        ]
    )

    gr.Markdown("""
    ## Cataract Screening — Result Interpretation

    This system categorizes eye images into one of four outcomes based on detected lens patterns.

    ---

    ### **Cataract Present**

    **What this means:**  
    Clear imaging evidence of cataract is detected. Lens opacity patterns are consistent with clinically significant cataract.

    **What to do:**  
    Clinical evaluation by an eye specialist is recommended.

    ---

    ### **Likely Early (Immature) Cataract**

    **What this means:**  
    Imaging patterns suggest early or immature cataract formation. Structural lens changes are present but not advanced.

    **What to do:**  
    Routine monitoring or clinical correlation is advised.

    ---

    ### **Likely Non-Cataract (Early Overlap)**

    **What this means:**  
    No definitive cataract is detected. Subtle lens patterns may overlap with very early cataract features or normal physiological variation.

    **What to do:**  
    No immediate action is required. Monitoring is recommended if symptoms develop.

    ---

    ### **No Cataract Detected**

    **What this means:**  
    The lens appearance is consistent with a healthy lens. No imaging evidence of cataract is observed.

    **What to do:**  
    No follow-up is required at this time.

    ---

    ### **Important Note**

    This tool is designed for **screening and educational purposes**.  
    It does not replace a clinical eye examination..
    """)

demo.launch()
