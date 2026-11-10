from io import BytesIO

import gradio as gr
import requests
import torch
from tqdm import tqdm
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
from torchvision.transforms import ToPILImage

from utils import (
    preprocess,
    prepare_mask_and_masked_image,
    recover_image,
    resize_and_crop,
)

gr.close_all()
topil = ToPILImage()

# The research implementation below is kept intact; this update focuses on
# the Gradio experience and safer local demo defaults.
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe_inpaint = pipe_inpaint.to("cuda")

GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 100
DEFAULT_SEED = 1234


def pgd(
    X,
    targets,
    model,
    criterion,
    eps=0.1,
    step_size=0.015,
    iters=40,
    clamp_min=0,
    clamp_max=1,
    mask=None,
):
    X_adv = X.clone().detach() + (torch.rand(*X.shape) * 2 * eps - eps).cuda()
    pbar = tqdm(range(iters))

    for i in pbar:
        actual_step_size = step_size - (
            step_size - step_size / 100
        ) / iters * i
        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean - targets).norm()
        pbar.set_description(
            f"Loss {loss.item():.5f} | step size: {actual_step_size:.4}"
        )

        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        if mask is not None:
            X_adv.data *= mask

    return X_adv


def get_target():
    target_url = (
        "https://www.rtings.com/images/test-materials/"
        "2015/204_Gray_Uniformity.png"
    )
    response = requests.get(target_url, timeout=30)
    response.raise_for_status()
    target_image = Image.open(BytesIO(response.content)).convert("RGB")
    return target_image.resize((512, 512))


def immunize_fn(init_image, mask_image):
    with torch.autocast("cuda"):
        mask, X = prepare_mask_and_masked_image(init_image, mask_image)
        X = X.half().cuda()
        mask = mask.half().cuda()

        targets = pipe_inpaint.vae.encode(
            preprocess(get_target()).half().cuda()
        ).latent_dist.mean

        adv_X = pgd(
            X,
            targets=targets,
            model=pipe_inpaint.vae.encode,
            criterion=torch.nn.MSELoss(),
            clamp_min=-1,
            clamp_max=1,
            eps=0.12,
            step_size=0.01,
            iters=200,
            mask=1 - mask,
        )

        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
        adv_image = topil(adv_X[0]).convert("RGB")
        return recover_image(
            adv_image, init_image, mask_image, background=True
        )


def run(image, prompt, seed, guidance_scale, num_inference_steps, immunize=False):
    if image is None:
        raise gr.Error("Upload an image or choose one of the examples first.")
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt describing the edit you want to test.")

    seed = DEFAULT_SEED if seed in (None, "") else int(seed)
    torch.manual_seed(seed)

    init_image = Image.fromarray(image["image"])
    init_image = resize_and_crop(init_image, (512, 512))
    mask_image = ImageOps.invert(
        Image.fromarray(image["mask"]).convert("RGB")
    )
    mask_image = resize_and_crop(mask_image, init_image.size)

    if immunize:
        immunized_image = immunize_fn(init_image, mask_image)

    image_edited = pipe_inpaint(
        prompt=prompt,
        image=init_image if not immunize else immunized_image,
        mask_image=mask_image,
        height=init_image.size[0],
        width=init_image.size[1],
        eta=1,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    image_edited = recover_image(image_edited, init_image, mask_image)

    if immunize:
        return [
            (immunized_image, "Protected input (research method)"),
            (image_edited, "Edit attempt after protection"),
        ]

    return [(image_edited, "Edit attempt without protection")]


CUSTOM_CSS = """
:root {
  --pg-ink: #101d36;
  --pg-muted: #64748b;
  --pg-blue: #2563eb;
  --pg-cyan: #38bdf8;
  --pg-line: #dce5f1;
}
.gradio-container {
  max-width: 1380px !important;
  margin: 0 auto !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.pg-shell { padding: 12px 0 28px; }
.pg-hero {
  position: relative;
  overflow: hidden;
  padding: 34px 38px;
  border: 1px solid rgba(148, 163, 184, .22);
  border-radius: 22px;
  color: #f8fafc;
  background: linear-gradient(118deg, #0b1730 0%, #142c55 58%, #075985 100%);
  box-shadow: 0 18px 50px rgba(15, 35, 70, .16);
}
.pg-hero:after {
  content: "";
  position: absolute;
  width: 330px; height: 330px;
  right: -100px; top: -170px;
  border: 1px solid rgba(186, 230, 253, .2);
  border-radius: 50%;
  box-shadow: 0 0 0 34px rgba(186,230,253,.035), 0 0 0 72px rgba(186,230,253,.025);
}
.pg-kicker {
  display: inline-flex; align-items: center; gap: 8px;
  margin-bottom: 14px; color: #bae6fd;
  font-size: 11px; font-weight: 750; letter-spacing: .16em;
  text-transform: uppercase;
}
.pg-dot { width: 7px; height: 7px; border-radius: 50%; background: #38bdf8; }
.pg-hero h1 {
  position: relative; z-index: 1;
  margin: 0 0 12px; color: #fff; font-size: clamp(30px, 4vw, 46px);
  line-height: 1.08; letter-spacing: -.04em; font-weight: 760;
}
.pg-hero p {
  position: relative; z-index: 1; max-width: 760px;
  margin: 0; color: #dbeafe; font-size: 15px; line-height: 1.75;
}
.pg-meta { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 9px; margin-top: 22px; }
.pg-meta a, .pg-meta span {
  color: #e0f2fe; text-decoration: none; font-size: 12px;
  border: 1px solid rgba(219,234,254,.25); border-radius: 999px;
  padding: 7px 12px; background: rgba(255,255,255,.07);
}
.pg-section-title { margin: 24px 0 10px; font-size: 18px; font-weight: 700; letter-spacing: -.02em; }
.pg-subtitle { color: var(--pg-muted); font-size: 13px; line-height: 1.65; }
.pg-panel {
  border: 1px solid var(--pg-line); border-radius: 16px;
  padding: 18px; background: var(--background-fill-primary);
}
.pg-note {
  border-left: 3px solid #38bdf8; padding: 11px 14px;
  border-radius: 0 10px 10px 0; background: rgba(56,189,248,.08);
  color: var(--body-text-color); font-size: 12px; line-height: 1.65;
}
.pg-footer {
  margin-top: 24px; padding: 18px 4px 4px; text-align: center;
  border-top: 1px solid var(--pg-line); color: var(--pg-muted);
  font-size: 12px; line-height: 1.8;
}
.pg-footer a { color: var(--pg-blue); text-decoration: none; }
button.primary {
  border-radius: 10px !important; font-weight: 700 !important;
}
button { transition: transform .16s ease, box-shadow .16s ease !important; }
button:hover { transform: translateY(-1px); }
@media (max-width: 700px) {
  .pg-hero { padding: 26px 22px; border-radius: 16px; }
  .pg-shell { padding-top: 0; }
}
"""


HEADER = """
<div class="pg-shell">
  <section class="pg-hero">
    <div class="pg-kicker"><span class="pg-dot"></span> IMAGE PROTECTION RESEARCH</div>
    <h1>PhotoGuard</h1>
    <p>
      Explore the research behind making unauthorized AI-powered image edits
      more difficult. Compare an edit attempt with and without the
      experimental protection method.
    </p>
    <div class="pg-meta">
      <a href="https://arxiv.org/abs/2302.06588" target="_blank">Research paper ↗</a>
      <a href="https://github.com/MadryLab/photoguard" target="_blank">Original research ↗</a>
      <a href="https://gradientscience.org/photoguard/" target="_blank">Research notes ↗</a>
      <span>Maintained fork · Utkarsh Jadhav</span>
    </div>
  </section>
</div>
"""

FOOTER = """
<div class="pg-footer">
  <strong>PhotoGuard</strong> · Research implementation based on the work
  of Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas and
  Aleksander Madry.<br>
  Fork documentation and interface maintained by
  <a href="https://github.com/uttax3007-lgtm" target="_blank">Utkarsh Jadhav</a>.
  Not a guarantee of image security or permanent protection.
</div>
"""

EXAMPLES = [
    ["./images/hadi_and_trevor.jpg", "man attending a wedding", "329357",
     GUIDANCE_SCALE, NUM_INFERENCE_STEPS, False],
    ["./images/trevor_2.jpg", "two men in prison", "329357",
     GUIDANCE_SCALE, NUM_INFERENCE_STEPS, False],
    ["./images/elon_2.jpg", "man in a metro station", "214213",
     GUIDANCE_SCALE, NUM_INFERENCE_STEPS, False],
]


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
    title="PhotoGuard | Image Protection Research",
) as demo:
    gr.HTML(HEADER)

    with gr.Row():
        with gr.Column(scale=7):
            gr.Markdown("## Run an experiment")
            gr.Markdown(
                "Upload an image, mark the area to preserve, and describe "
                "the edit you want to test. The mask marks the region that "
                "should remain unchanged during inpainting.",
                elem_classes=["pg-subtitle"],
            )
        with gr.Column(scale=5):
            gr.HTML(
                '<div class="pg-note"><strong>Research preview</strong><br>'
                'This demo runs the original experimental method. Outputs '
                'depend on the model and settings; protection is not '
                'absolute and should not be treated as a security guarantee.'
                '</div>'
            )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["pg-panel"]):
            gr.Markdown("### 01 · Prepare your image")
            imgmask = gr.ImageMask(
                label="Image and preserve mask",
                type="numpy",
                sources=["upload", "clipboard"],
                height=420,
            )
            prompt = gr.Textbox(
                label="Edit prompt",
                placeholder="Describe the change, e.g. add a red hat",
                lines=2,
            )
            seed = gr.Textbox(
                label="Random seed",
                placeholder=str(DEFAULT_SEED),
                value=str(DEFAULT_SEED),
                info="Keep the same seed when comparing runs.",
            )

            with gr.Accordion("Advanced generation settings", open=False):
                scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=25.0,
                    value=GUIDANCE_SCALE,
                    step=0.1,
                )
                num_steps = gr.Slider(
                    label="Inference steps",
                    minimum=10,
                    maximum=250,
                    value=NUM_INFERENCE_STEPS,
                    step=5,
                )

            immunize = gr.Checkbox(
                label="Apply experimental PhotoGuard method before editing",
                value=False,
                info="Enable this to compare the protected-image workflow.",
            )
            b1 = gr.Button("Run experiment", variant="primary", size="lg")

        with gr.Column(scale=7, elem_classes=["pg-panel"]):
            gr.Markdown("### 02 · Review the result")
            genimages = gr.Gallery(
                label="Generated outputs",
                show_label=False,
                elem_id="gallery",
            ).style(grid=[1, 2], height="auto")
            gr.Markdown(
                "The comparison is intended for research and demonstration. "
                "Visual results alone do not establish that an image is "
                "protected against other editing systems.",
                elem_classes=["pg-subtitle"],
            )

    b1.click(
        run,
        [imgmask, prompt, seed, scale, num_steps, immunize],
        [genimages],
    )

    with gr.Accordion("How to use this demo", open=False):
        gr.Markdown(
            """
            1. Upload an image or select an example.
            2. Use the mask brush to mark the region you want to preserve
               (for example, a person's face).
            3. Enter a prompt describing the edit.
            4. Run once with the protection option disabled to see the
               baseline edit attempt.
            5. Enable the experimental PhotoGuard option and run again to
               compare the protected-image workflow.

            For a meaningful comparison, keep the prompt and seed unchanged.
            The method is experimental and may not generalize to other models
            or image-editing pipelines.
            """
        )

    with gr.Accordion("Example inputs", open=False):
        gr.Examples(
            examples=EXAMPLES,
            inputs=[imgmask, prompt, seed, scale, num_steps, immunize],
            outputs=[genimages],
            fn=run,
            cache_examples=False,
        )

    with gr.Accordion("Research and attribution", open=False):
        gr.Markdown(
            """
            **Raising the Cost of Malicious AI-Powered Image Editing**  
            Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas and
            Aleksander Madry.

            [Paper (arXiv:2302.06588)](https://arxiv.org/abs/2302.06588) ·
            [Original MadryLab repository](https://github.com/MadryLab/photoguard)

            Please cite the original paper when using this implementation
            in academic work. See the repository's MIT license for terms.
            """
        )

    gr.HTML(FOOTER)


if __name__ == "__main__":
    # Keep the demo local by default. Set share=True explicitly only when
    # you intentionally want Gradio to create a public share link.
    demo.launch(server_name="127.0.0.1", share=False, server_port=7860)
