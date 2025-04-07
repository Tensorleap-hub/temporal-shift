import onnx
import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn

from ops.models import TSN

class Permute(nn.Module):
    """
    A simple wrapper module that permutes tensor dimensions.

    Args:
        *dims (int): The order of dimensions to permute the input tensor to.
    """
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

# Configuration
model_path = "pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth"
onnx_path = "tsm.onnx"
input_tensor = torch.randn(1, 3, 8, 256, 256)  # Dummy input for export

# Initialize TSN model
model = TSN(
    num_class=400,
    num_segments=8,
    modality="RGB",
    base_model="resnet50",
    consensus_type="avg",
    img_feature_dim=256,
    pretrain="imagenet",
    is_shift=True,
    shift_div=8,
    shift_place="blockres",
    non_local='_nl' in model_path
)

# Apply temporal pooling if specified in model path
if 'tpool' in model_path:
    from ops.temporal_shift import make_temporal_pool
    make_temporal_pool(model.base_model, 8)

# Load and clean checkpoint
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in checkpoint.items()}

# Rename keys to match model definition
replace_dict = {
    'base_model.classifier.weight': 'new_fc.weight',
    'base_model.classifier.bias': 'new_fc.bias',
}
for old_key, new_key in replace_dict.items():
    if old_key in base_dict:
        base_dict[new_key] = base_dict.pop(old_key)

model.load_state_dict(base_dict)

# Wrap model with dimension permutation for compatibility
model = nn.Sequential(
    Permute(0, 2, 1, 3, 4),  # (B, C, F, H, W) -> (B, F, C, H, W)
    model
)
model.eval()

# Export the model to ONNX format
torch.onnx.export(
    model.float(),
    input_tensor,
    onnx_path,
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Validate ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Run inference using ONNX Runtime
ort_session = ort.InferenceSession(onnx_path)
onnx_input = input_tensor.numpy()
onnx_outputs = ort_session.run(None, {'input': onnx_input})
onnx_output = onnx_outputs[0]

# Run inference using PyTorch
with torch.no_grad():
    torch_output = model(input_tensor).numpy()

# Compare ONNX and PyTorch outputs
np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-03, atol=1e-05)
print("SUCCESS: The outputs from PyTorch and ONNX match!")