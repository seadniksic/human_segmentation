import torch_tensorrt
from custom_models.unet_batchnorm import UNet
import torch

model_path = "checkpoints\\long_trains\\UNet_Standard_BatchNorm3.pt"

model_save = torch.load(model_path)
model = UNet() # torch module needs to be in eval (not training) mode

model.load_state_dict(model_save['model_state_dict'])

model.eval()

inputs = [
    torch_tensorrt.Input(
        min_shape=[1, 3, 256, 256],
        opt_shape=[1, 3, 256, 256],
        max_shape=[1, 3, 256, 256],
        dtype=torch.half,
    )
]
enabled_precisions = {torch.float, torch.half}  # Run with fp16

trt_ts_module = torch_tensorrt.compile(
    model, inputs=inputs, enabled_precisions=enabled_precisions
)

torch.jit.save(trt_ts_module, "compiled_tensorrt\\UNet.ts")