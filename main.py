# main.py
from diffusion.pipeline import load_pipeline
from diffusion.generate import generate_story_images
from prompts.prompt_util import build_prompts
from unet.unet_controller import apply_unet_controls
import torch
import os

torch.set_num_threads(os.cpu_count())
torch.backends.mkldnn.enabled = True

MODEL_ID = "Lykon/DreamShaper-7"
DEVICE = "cpu"

print(" Loading diffusion pipeline (ONCE per process)")

#  Load pipeline once
PIPELINE = load_pipeline(MODEL_ID, DEVICE)
apply_unet_controls(PIPELINE)

def run_pipeline(
    id_prompt,
    frame_prompts,
    negative_prompt,
    seed,
    height,
    width,
):
    if not frame_prompts:
        raise ValueError("Frame prompts cannot be empty")

    final_prompts = build_prompts(id_prompt, frame_prompts)

    story_path, output_dir, frame_paths = generate_story_images(
        pipe=PIPELINE,
        prompts=final_prompts,
        negative_prompt=negative_prompt,
        seed=seed,
        height=height,
        width=width,
    )

    return story_path, output_dir,  frame_paths


# -------------------------------------------------
#  RUN STORY GENERATION WHEN main.py IS EXECUTED
# -------------------------------------------------
if __name__ == "__main__":
    print(" Running demo story generation from main.py")

    ID_PROMPT = (
        "A kind teacher and a student in a classroom, consistent characters"
    )

    FRAME_PROMPTS = [
        "The teacher explaining a lesson",
        "The student listening carefully",
        "The student asking a question",
        "The teacher smiling and answering",
    ]

    NEGATIVE_PROMPT = (
        "extra people, distorted faces, low quality, blurry"
    )

    story_path, output_dir = run_pipeline(
        id_prompt=ID_PROMPT,
        frame_prompts=FRAME_PROMPTS,
        negative_prompt=NEGATIVE_PROMPT,
        seed=42,
        height=320,
        width=320,
    )

    print("Story generated successfully")
    print("Output folder:", output_dir)
    print("Story image:", story_path)
