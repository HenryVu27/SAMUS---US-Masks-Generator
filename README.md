# SAMUS bladder mask generation

Inference and mask-refinement scripts that run the pretrained **SAMUS** model on bladder ultrasound images.

This is **not** the SAMUS repository. SAMUS is by Lin et al.; the model, the paper, and the pretrained weights are theirs:

- Paper: [SAMUS: Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation](https://arxiv.org/pdf/2309.06824.pdf)
- Official code and weights: https://github.com/xianlin7/SAMUS

## What is here

| File | Purpose |
|---|---|
| `infer_bladder_images.py` | Runs pretrained SAMUS over a directory of bladder ultrasound frames and writes masks |
| `infer_bladder_images_refined.py` | Same, with the refinement pass applied |
| `mask_refiner.py` | Post-processing to clean up raw predicted masks |
| `example_mask_refinement.py` | Minimal usage example |

See [`README_mask_refinement.md`](README_mask_refinement.md) for the refinement details.

## Usage

Clone the [official SAMUS repo](https://github.com/xianlin7/SAMUS) and download their pretrained checkpoint first, then:

```bash
pip install -r requirements.txt
python infer_bladder_images.py --help
```

## License

See [`LICENSE`](LICENSE). SAMUS itself is licensed by its authors; check the upstream repository before redistributing weights or derived models.
