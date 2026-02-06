# CIFAR-10 PyTorch Example

Small tutorial-style PyTorch project that trains a simple CNN on CIFAR-10.

Quick start:

```bash
python "train.py"
```

Notes:
- The original file name contained a space; this repository uses `train.py`.
- If running on Windows or in interactive shells, DataLoader uses `num_workers=0` to avoid subprocess issues.
- A sample image is saved to `sample.png`. The trained model is saved to `model.pth`.

Dependencies are listed in `requirements.txt`.
