# Face Inpainting
implementation of a deep network that can fill holes in images of faces. The top level files are just the model and obstruiction detection code. It can be run with `python main.py --image <path to obstructed image> --model <path to model weights>`. No model weights are provided as they are too large for git. 

The notebook used for training can be found in the traing folder. It assumes that UTKFace and CelebA-HQ zipfiles are present (those are also too large for github).

# Dependencies 

 - A GPU with 8GB of VRAM (16GB if training)
 - Pytorch with Cuda
 - numpy
 - PILLOW
 - OpenCV 2

# References

[Nvidia Image Inpainting for Irregular Holes Using Partial Convolutions paper](https://arxiv.org/pdf/1804.07723.pdf)
