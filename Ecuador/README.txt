main.py creates an adversarial image that makes a MobileNetV3 Large classifier recognize a mountain bicycle as a moped (ImageNet class 665) with at least 99% confidence, while changing each RGB value by no more than 1. The output keeps the original image dimensions and is saved as an RGB PNG under 1 MiB.

The script does four steps:
1. Loads the bicycle image and the pretrained MobileNetV3 Large classifier from challenge_model.py. It converts the image into a tensor and sets the target class to 665 (moped).
2. Uses a targeted Projected Gradient Descent (PGD) attack. It calculates gradients to increase the model's confidence in the moped class, while restricting every pixel change to ±1/255. If needed, it tries different starting perturbations.
3. Checks the modified image after rounding RGB values to integers and keeps the version with the highest moped confidence. The attack stops early when confidence reaches 99.5% or when the time limit approaches.
4. Saves the result as a PNG and verifies its dimensions, maximum RGB difference, and file size. On the public training image, the attack achieved 99.5481% moped confidence in 0.87 seconds, exceeding the required 99% threshold. Performance on the unseen test image is not guaranteed.

Usage:
Put main.py, challenge_model.py, train-input.png and classifier/mobilenet_v3_large-5c1a4163.pth in the challenge folder, then run:

py -3.13 main.py --input train-input.png --output candidate-output.png

For official evaluation, run:

python main.py

It reads test-input.png and writes test-output.png automatically. Requires Python 3.13.5, torch==2.9.1, torchvision==0.24.1, Pillow==11.3.0 and numpy==2.2.6. The script runs on CPU without internet access.