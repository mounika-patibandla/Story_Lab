def build_prompts(id_prompt, frame_prompts):
    return [f"{id_prompt}, {fp}" for fp in frame_prompts]
