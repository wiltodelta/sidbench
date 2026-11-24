# RAIW v10 ProGAN pilot

This experiment uses the [Wang2020 ProGAN test set](https://huggingface.co/datasets/sywang/CNNDetection/tree/2dd6d95d4333188b8616f3d34970ea800ca1e339), which SIDBench lists in `dataset/dataset_paths.py`. The archive contains 200 real and 200 generated images in each of 20 categories. The script takes four images from each category and label, ordered by the SHA-256 of the member path. It evaluates 80 real and 80 generated images using the original RGB pixels, with no added JPEG compression or blur. The CSV records the exact member paths and SHA-256 of each image.

The experiment uses SIDBench's metric functions from commit `b33f51c7c25868a69cb7ba7d1fdc582e243acbf0`. The sample is a pilot. The full ProGAN split and possible overlap with RAIW training data were outside its scope.

## Reproduce

The recorded run used Python 3.12.16 on macOS arm64, `remove-ai-watermarks` 0.72.0 and ONNX Runtime 1.32.0 on CPU. The local requirements file pins the entire Python environment. Run these commands from the repository root:

```sh
pilot_dir="$(mktemp -d)"
uv venv --python 3.12.16 "$pilot_dir/venv"
uv pip sync --python "$pilot_dir/venv/bin/python" experiments/raiw-progan/requirements.txt
curl -fL 'https://huggingface.co/datasets/sywang/CNNDetection/resolve/2dd6d95d4333188b8616f3d34970ea800ca1e339/progan_testset.zip' \
  -o "$pilot_dir/progan_testset.zip"
curl -fL 'https://huggingface.co/wiltodelta/raiw-source-classify/resolve/5f3d7294cad4350d5511e3fb1c1f8eacb8033f59/multi-provider-source-v10.npz' \
  -o "$pilot_dir/multi-provider-source-v10.npz"
"$pilot_dir/venv/bin/python" -B experiments/raiw-progan/run.py \
  --archive "$pilot_dir/progan_testset.zip" \
  --weights "$pilot_dir/multi-provider-source-v10.npz" \
  --output "$pilot_dir/results"
```

The runner checks both file hashes and passes the verified weights path explicitly to RAIW. Once the environment and files are in place, inference works offline. To enforce this on macOS, put `sandbox-exec -p '(version 1)(allow default)(deny network*)'` before the Python command.

[Predictions](results/predictions.csv) include the raw RAIW score used for ROC AUC and Average Precision. [Metrics](results/metrics.json) show the confusion matrices and accuracy for SIDBench's `score > 0.5` rule and for the classifier's native binary decision. The native decision comes from the classifier API because v10 can combine several decision branches.

The recorded run produced 158 correct decisions out of 160. Both rules made the same one false-positive and one false-negative error. ROC AUC was 0.99875 and Average Precision was 0.99871. A second run in a fresh environment installed from the same requirements produced byte-identical CSV and JSON files with networking denied. A file with the wrong checkpoint hash was rejected before inference.
