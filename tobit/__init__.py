import os
import torch.utils.cpp_extension
from tobit.standard import KalmanFilter
from tobit.tobit import TobitKalmanFilter


def load_cpp():
    torch.utils.cpp_extension.load(
        name="tkf",
        sources=[os.path.join(os.path.dirname(__file__), "cpp", "op.cpp")],
        extra_ldflags=[],
        is_python_module=False,
        verbose=False,
    )