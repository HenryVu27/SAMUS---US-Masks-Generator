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

## How it works

The pipeline writes one binary mask per input frame. `infer_bladder_images.py` resizes each grayscale frame to 256 by 256 pixels and repeats it across three channels. SAMUS, with its ViT-B (Vision Transformer, base) image encoder, gets a single positive point prompt at the image center, so the script relies on the bladder sitting near the middle of each frame. The script thresholds the sigmoid output at 0.5 and resizes the mask back to the frame's original size. It saves the result as `<frame>_mask.png`.

The refinement pass exists to clean up raw masks that come out choppy or split into several disconnected regions. `infer_bladder_images_refined.py` keeps the probability map instead of thresholding it. It resizes the map bilinearly to the original frame size and passes it to `MaskRefiner` in `mask_refiner.py`. With the script's defaults, refinement runs these steps in order.

1. Pick a threshold with Otsu's method on a copy of the probability map blurred with a 5 by 5 Gaussian kernel, then binarize the map with it.
2. Drop connected components smaller than 150 pixels.
3. Fill holes of up to 75 pixels. Larger holes stay open.
4. Run a morphological opening and then a closing with a 3 by 3 elliptical kernel.
5. Keep only the largest connected component.
6. Smooth the edges with a Gaussian filter (sigma 0.5) and threshold again at 0.5.

The refined script saves `<frame>_mask_refined.png`. The `--save_probability_maps` and `--save_unrefined` flags also save the raw probability maps and plain 0.5-threshold masks for comparison. With `--create_comparison`, it writes `refinement_comparison.png`, which places up to five masks from `segmentation_masks/`, the default output folder of `infer_bladder_images.py`, next to their refined versions. `mask_refiner.py` also runs on its own over a folder of saved masks.

The masks have not been scored against ground-truth labels.

## Usage

Clone the [official SAMUS repo](https://github.com/xianlin7/SAMUS) and download their pretrained checkpoint first, then:

```bash
pip install -r requirements.txt
python infer_bladder_images.py --help
```

## License

See [`LICENSE`](LICENSE). SAMUS itself is licensed by its authors; check the upstream repository before redistributing weights or derived models.
