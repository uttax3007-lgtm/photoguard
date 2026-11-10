# PhotoGuard

Research code for protecting images against certain forms of malicious AI-powered editing.

This repository is a personal fork of the PhotoGuard research project from MadryLab. It keeps the original research implementation and attribution intact, with repository documentation and maintenance notes adapted for this fork.

**Maintainer:** Utkarsh Jadhav  
**GitHub:** [@uttax3007-lgtm](https://github.com/uttax3007-lgtm)  
**Upstream project:** [MadryLab/photoguard](https://github.com/MadryLab/photoguard)

> This is a research implementation, not a guarantee that an image cannot be edited or manipulated. Results depend on the model, pipeline, settings, and image being tested.

## About the research

PhotoGuard explores image-protection techniques intended to make unauthorized AI-powered edits more difficult. The original work studies attacks against image-editing pipelines based on diffusion models, including image-to-image and inpainting workflows.

The implementation and research are based on:

**Raising the Cost of Malicious AI-Powered Image Editing**  
Hadi Salman*, Alaa Khaddaj*, Guillaume Leclerc*, Andrew Ilyas, Aleksander Madry

- Paper: https://arxiv.org/abs/2302.06588
- Research write-up: https://gradientscience.org/photoguard/
- Original repository: https://github.com/MadryLab/photoguard
- Hosted demo: https://huggingface.co/spaces/hadisalman/photoguard

Please cite the paper when using the research in academic work:

```bibtex
@article{salman2023raising,
  title={Raising the Cost of Malicious AI-Powered Image Editing},
  author={Salman, Hadi and Khaddaj, Alaa and Leclerc, Guillaume and Ilyas, Andrew and Madry, Aleksander},
  journal={arXiv preprint arXiv:2302.06588},
  year={2023}
}
```

## Getting started

### Requirements

The original implementation uses Python and the Hugging Face Diffusers ecosystem. Model execution may require a compatible GPU and a correctly installed PyTorch build.

The dependency versions in this repository reflect the research code and may need adjustment for newer Python, CUDA, PyTorch, or Diffusers releases.

### Installation

1. Clone this fork:

   ```bash
   git clone https://github.com/uttax3007-lgtm/photoguard.git
   cd photoguard
   ```

2. Create and activate an environment:

   ```bash
   conda create -n photoguard python=3.10
   conda activate photoguard
   ```

3. Install the project dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. If a workflow downloads gated or authenticated Hugging Face models, configure your Hugging Face account and authenticate using the CLI supported by your installed Hugging Face Hub version.

## Running the interactive demo

The Gradio application is located in the `demo/` directory.

```bash
conda activate photoguard
cd demo
python app.py
```

The hosted research demo is also available on [Hugging Face Spaces](https://huggingface.co/spaces/hadisalman/photoguard). For reproducible experiments, prefer running the code locally with documented model and dependency versions.

## Research workflows

The notebooks in `notebooks/` walk through the experiments from the paper:

- `generating_fake_images.ipynb` — generating example images.
- `demo_simple_attack_img2img.ipynb` — image-to-image protection experiments.
- `demo_simple_attack_inpainting.ipynb` — inpainting experiments using the simpler encoder-based approach.
- `demo_complex_attack_inpainting.ipynb` — end-to-end diffusion attack experiments for inpainting.

Run the notebooks in order where one experiment depends on outputs produced by an earlier notebook. Check each notebook's cells for model downloads, GPU requirements, and expected input files.

## Project layout

```text
photoguard/
├── assets/       # Figures and demo images
├── demo/         # Gradio interactive demo
├── notebooks/    # Research and experiment notebooks
├── LICENSE
├── README.md
└── requirements.txt
```

## Limitations and responsible use

- PhotoGuard is an experimental research approach. It does not provide universal or permanent protection against image editing.
- Do not treat a successful experiment as proof that an image is safe from manipulation.
- Model updates, preprocessing, image transformations, and different editing pipelines can change the results.
- Use the techniques for research, evaluation, and the protection of image owners. Do not use this project to interfere with legitimate creative work or access to content.
- Do not upload private or sensitive images to services you do not trust.

## Attribution and changes in this fork

The underlying research, original implementation, and associated materials are credited to the authors and MadryLab. This fork is maintained by **Utkarsh Jadhav**. Changes made specifically in this fork are separate from the original upstream work; see the commit history for the exact changes.

If you build on the research, please cite the paper above and retain the applicable notices in the [MIT license](LICENSE).

## Contributing

This is a personal research fork. Before proposing changes, please read [CONTRIBUTING.md](CONTRIBUTING.md). Keep experimental changes reproducible and clearly distinguish new work from the original implementation.

## License

The project is distributed under the MIT License. The original MadryLab copyright notice is retained, and a separate notice covers modifications made in this fork. See [LICENSE](LICENSE) for the full terms.
