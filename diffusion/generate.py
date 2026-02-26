import os
from datetime import datetime
from PIL import Image
import torch

NUM_STEPS = 20
GUIDANCE_SCALE = 5.8


def generate_story_images(
    pipe,
    prompts,
    negative_prompt,
    seed,
    height,
    width,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"story_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    #  Single generator
    generator = torch.manual_seed(seed)

    #  BATCH generation (one pipe call only)
    results = pipe(
        prompts,
        negative_prompt=[negative_prompt] * len(prompts),
        generator=generator,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=height,
        width=width,
    )

    images = results.images
    frame_paths = []

    # Save images
    for i, image in enumerate(images, start=1):
        frame_path = os.path.join(output_dir, f"frame_{i}.png")
        image.save(frame_path)
        frame_paths.append(frame_path)

    # Create horizontal strip
    total_width = width * len(images)
    story_strip = Image.new("RGB", (total_width, height))

    x_offset = 0
    for img in images:
        story_strip.paste(img, (x_offset, 0))
        x_offset += width

    story_path = os.path.join(output_dir, "story_strip.png")
    story_strip.save(story_path)

    return story_path, output_dir, frame_paths
