def apply_unet_controls(pipe):
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
