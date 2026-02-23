import sys
import os
import math
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from main import run_pipeline
import ollama


NEGATIVE_PROMPT = (
    "extra people, distorted faces, asymmetrical eyes, different eye color, "
    "low quality, blurry"
)

# -------------------------------------------------
# 🔥 STRONG CONSISTENCY ENHANCER
# -------------------------------------------------
def enhance_for_consistency(id_prompt, short_prompts):

    enhanced = []

    for short in short_prompts:
        full_prompt = (
            f"{id_prompt}. "
            f"In this scene: {short}. "
            "same face, same eye shape, same hairstyle, "
            "consistent character design, cinematic lighting, soft shading."
        )
        enhanced.append(full_prompt)

    return enhanced


# -------------------------------------------------
#  SAFE CAPTION WRAP
# -------------------------------------------------
def add_caption(image_path, text):

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    try:
        font = ImageFont.truetype("arial.ttf", size=int(h * 0.05))
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)

    # Wrap text properly
    max_width = int(w * 0.9)
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    line_height = draw.textbbox((0, 0), "A", font=font)[3]
    padding = 20
    caption_height = line_height * len(lines) + padding * 2

    new_img = Image.new("RGB", (w, h + caption_height), (255, 255, 255))
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)

    y_text = h + padding
    for line in lines:
        draw.text(
            (w // 2, y_text),
            line,
            fill=(0, 0, 0),
            font=font,
            anchor="ma"
        )
        y_text += line_height

    new_img.save(image_path)


# -------------------------------------------------
#  BETTER GRID LOGIC
# -------------------------------------------------
def calculate_grid(n):
    if n <= 3:
        return n, 1
    elif n == 4:
        return 2, 2
    elif 5 <= n <= 6:
        return 3, 2
    elif 7 <= n <= 9:
        return 3, 3
    else:
        return 4, 3


def create_grid(image_paths):

    images = [Image.open(p).convert("RGB") for p in image_paths]
    n = len(images)

    cols, rows = calculate_grid(n)

    w, h = images[0].size
    spacing = 40  # bigger spacing

    grid_width = cols * w + (cols - 1) * spacing
    grid_height = rows * h + (rows - 1) * spacing

    grid_img = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        x = col * (w + spacing)
        y = row * (h + spacing)

        grid_img.paste(img, (x, y))

    os.makedirs("results", exist_ok=True)
    final_path = os.path.join("results", "final_grid.png")
    grid_img.save(final_path, quality=95)

    return final_path


# -------------------------------------------------
# 🚀 MAIN GENERATION
# -------------------------------------------------
def run_generation(id_prompt, frame_text, num_frames,
                   height, width, seed):

    short_prompts = [
        line.strip()
        for line in frame_text.split("\n")
        if line.strip()
    ][:int(num_frames)]

    if len(short_prompts) < int(num_frames):
        return None, "Please provide enough frame prompts."

    enhanced_prompts = enhance_for_consistency(id_prompt, short_prompts)

    # Use same base seed for better identity stability
    story_path, output_dir, frame_paths = run_pipeline(
        id_prompt="",
        frame_prompts=enhanced_prompts,
        negative_prompt=NEGATIVE_PROMPT,
        seed=int(seed),
        height=int(height),
        width=int(width),
    )

    # Add captions after generation
    for img_path, caption in zip(frame_paths, short_prompts):
        add_caption(img_path, caption)

    grid_path = create_grid(frame_paths)

    return grid_path, output_dir
    # -------------------------------------------------
#  LLM STORY ASSISTANT
# -------------------------------------------------
# -------------------------------------------------
#  LLM STORY ASSISTANT (Gradio New Format)
# -------------------------------------------------
def generate_story_idea(user_input, chat_history, frame_count):

    if chat_history is None:
        chat_history = []

    system_prompt = f"""
You are a professional comic story prompt generator.

Output format strictly:

Identity Prompt: <detailed character description>

Then EXACTLY {frame_count} numbered scene prompts.

No explanation.
No extra text.
Follow numbering strictly.
"""

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    reply = response["message"]["content"]

    # ✅ NEW MESSAGE FORMAT
    chat_history.append({
        "role": "user",
        "content": user_input
    })

    chat_history.append({
        "role": "assistant",
        "content": reply
    })

    return "", chat_history

# -------------------------------------------------
# 📥 Extract Identity + Frames From Chat
# -------------------------------------------------
def extract_story_elements(chat_history):

    if not chat_history:
        return "", ""

    last_reply = chat_history[-1][1]
    lines = last_reply.split("\n")

    identity = ""
    frames = []

    for line in lines:
        if line.startswith("Identity Prompt"):
            identity = line.split(":", 1)[1].strip()

        elif line.strip().startswith(tuple(str(i)+"." for i in range(1, 20))):
            frames.append(line.split(".", 1)[1].strip())

    return identity, "\n".join(frames)

# -------------------------------------------------
# 🎨 UI
# -------------------------------------------------
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    body {
        background: linear-gradient(135deg, #141e30, #243b55);
    }

    .gradio-container {
        max-width: 100% !important;
        padding-left: 120px !important;
        padding-right: 120px !important;
    }

    /* Main Title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 40px;
        color: #2196f3 !important;
    }

    /* Section Headings */
    .section-heading {
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        margin-top: 40px;
        margin-bottom: 15px;
        color: #ffcc00 !important;
    }

    /* Chatbot Card */
    .chat-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        margin-bottom: 40px;
    }

    textarea {
        font-size: 17px !important;
    }
    """
) as demo:

    # ================= MAIN TITLE =================
    gr.HTML('<div class="main-title">📖 AI Comic Story Generator</div>')

    # ================= STORY ASSISTANT =================
    gr.HTML('<div class="section-heading">Story Assistant</div>')
    gr.HTML('<div class="chat-card">')

    chatbot = gr.Chatbot(height=400)

    msg = gr.Textbox(
        placeholder="Ask for a story idea..."
    )

    with gr.Row():
        fill_btn = gr.Button("Use Idea")
        clear_btn = gr.Button("Clear")

    gr.HTML('</div>')

    # ================= IDENTITY PROMPT =================
    gr.HTML('<div class="section-heading">Identity Prompt</div>')

    id_prompt = gr.Textbox(
        placeholder="Describe your main character in detail...",
        lines=7,
        show_label=False
    )

    # ================= FRAME PROMPTS =================
    gr.HTML('<div class="section-heading">Frame Prompts</div>')

    frame_prompts = gr.Textbox(
        placeholder="Write frame descriptions here (one per line)...",
        lines=12,
        show_label=False
    )

    # ================= SETTINGS =================
    gr.HTML('<div class="section-heading">Settings</div>')

    num_frames = gr.Slider(
        1, 12,
        value=6,
        step=1,
        label="Number of Frames"
    )

    with gr.Row():
        width = gr.Slider(256, 512, value=384, step=64, label="Width")
        height = gr.Slider(256, 512, value=384, step=64, label="Height")

    seed = gr.Number(value=42, label="Seed")

    generate_btn = gr.Button(" Generate Story", size="lg", variant="primary")

    # ================= GENERATED IMAGE =================
    gr.HTML('<div class="section-heading">Generated Story</div>')

    story_image = gr.Image(
        type="filepath",
        height=900,
        show_label=False
    )

    output_dir = gr.Textbox(show_label=False)

    # Existing image generation
    generate_btn.click(
      run_generation,
      inputs=[id_prompt, frame_prompts, num_frames, height, width, seed],
      outputs=[story_image, output_dir]
    )

# Chatbot
    msg.submit(
    generate_story_idea,
    [msg, chatbot, num_frames],
    [msg, chatbot]
)
    clear_btn.click(lambda: None, None, chatbot)

# Auto-fill prompts
    fill_btn.click(
    extract_story_elements,
    chatbot,
    [id_prompt, frame_prompts]
)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
)