import torch

# initialize 2 random tensors
x = torch.rand(5, 3)
y = torch.rand(5, 3)
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!  Training on GPU ...")
    # add x with y, using gpu
    x = x.cuda()
    y = y.cuda()
    z = x + y
else:
    print("CUDA is not available.  Training on CPU ...")
    z = x + y

print(z)
