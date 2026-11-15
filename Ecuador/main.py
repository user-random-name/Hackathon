import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from challenge_model import load_model, normalize_batch

TARGET = 665
THRESHOLD = 0.99
EPS = 1.0 / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='test-input.png')
    parser.add_argument('--output', default='test-output.png')
    parser.add_argument('--model', default='classifier/mobilenet_v3_large-5c1a4163.pth')
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(31415)
    start = time.monotonic()
    original_rgb = np.asarray(Image.open(args.input).convert('RGB'), dtype=np.uint8).copy()
    original = torch.from_numpy(original_rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255)
    model = load_model(args.model)
    goal = torch.tensor([TARGET])
    low = (original - EPS).clamp(0.0, 1.0)
    high = (original + EPS).clamp(0.0, 1.0)
    best_score = -1.0
    best_pixels = original_rgb.copy()

    @torch.no_grad()
    def verify(x):
        nonlocal best_score, best_pixels
        pixels = x.mul(255).round().clamp(0, 255).to(torch.uint8)
        actual = pixels.float().div(255)
        score = model(normalize_batch(actual)).softmax(dim=1)[0, TARGET].item()
        if score > best_score:
            best_score = score
            best_pixels = pixels[0].permute(1, 2, 0).contiguous().numpy().copy()
        return score

    verify(original)
    if best_score < THRESHOLD:
        # Start at the original image, then try alternative random starts if needed.
        for restart in range(7):
            if time.monotonic() - start > 46:
                break
            if restart == 0:
                x = original.clone()
            elif restart == 1:
                x = (original + (torch.rand_like(original) * 2 - 1) * EPS).clamp(low, high)
            elif restart == 2:
                x = low.clone()
            elif restart == 3:
                x = high.clone()
            elif restart == 4:
                x = torch.from_numpy(best_pixels.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255)
            else:
                x = (original + (torch.rand_like(original) * 2 - 1) * EPS).clamp(low, high)

            for i in range(250):
                if time.monotonic() - start > 48:
                    break
                x = x.detach().requires_grad_(True)
                output = model(normalize_batch(x))
                loss = F.cross_entropy(output, goal)
                gradient, = torch.autograd.grad(loss, x)
                with torch.no_grad():
                    # Projected gradient descent on the exact L-infinity constraint.
                    step = (0.30 if restart < 4 else 0.18) * EPS
                    x = (x - step * gradient.sign()).clamp(low, high)

                if i % 5 == 4 or i == 0:
                    score = verify(x)
                    if score >= 0.995:
                        break
            if best_score >= 0.995:
                break

    result = Image.fromarray(best_pixels, 'RGB')
    result.save(args.output, 'PNG', optimize=True, compress_level=9)
    # Validate the actual saved artifact and the budget (not float tensors).
    saved = np.asarray(Image.open(args.output).convert('RGB'), dtype=np.int16)
    assert saved.shape == original_rgb.shape
    assert np.abs(saved - original_rgb.astype(np.int16)).max() <= 1
    assert os.path.getsize(args.output) <= 1024 * 1024
    print(f'Moped confidence: {best_score:.6f}; PNG: {args.output}; time: {time.monotonic()-start:.2f}s')


if __name__ == '__main__':
    main()
