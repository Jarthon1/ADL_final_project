import torch

print("Version of torch:nvcc --version", torch.__version__)
print("Cuda is available?",torch.cuda.is_available())
print("Version of cuda", torch.version.cuda)
n_gpu = torch.cuda.device_count()
print("There are {} total gpus on this machine".format(n_gpu))
