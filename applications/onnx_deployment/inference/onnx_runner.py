"""Lightweight ONNX Runtime inference wrappers."""
import numpy as np
import onnxruntime as ort


class OnnxRunner:
    """Single-model ONNX Runtime inference wrapper."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Args:
            model_path: path to .onnx file
            device: "cpu" or "gpu"
        """
        if device == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.model_path = model_path

        # Detect expected input dtype (handles FP16 models)
        input_type = self.session.get_inputs()[0].type
        if "float16" in input_type:
            self._input_dtype = np.float16
        else:
            self._input_dtype = np.float32

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            input_array: numpy array matching model input shape

        Returns:
            output numpy array
        """
        if input_array.dtype != self._input_dtype:
            input_array = input_array.astype(self._input_dtype)
        return self.session.run([self.output_name], {self.input_name: input_array})[0]

    def info(self):
        """Print model input/output metadata."""
        print(f"Model: {self.model_path}")
        for inp in self.session.get_inputs():
            print(f"  Input:  {inp.name} shape={inp.shape} dtype={inp.type}")
        for out in self.session.get_outputs():
            print(f"  Output: {out.name} shape={out.shape} dtype={out.type}")


class StudentRunner:
    """Composite runner: Adaptation Module -> Base Policy (two-stage inference)."""

    def __init__(self, adaptation_path: str, policy_path: str, device: str = "cpu"):
        self.adaptation = OnnxRunner(adaptation_path, device)
        self.policy = OnnxRunner(policy_path, device)

    def __call__(self, obs_history: np.ndarray, obs_current: np.ndarray) -> np.ndarray:
        """Run full Student inference pipeline.

        Args:
            obs_history: (batch, 1350) flattened observation history
            obs_current: (batch, 27) current observation

        Returns:
            action: (batch, 8) action output
        """
        latent_z = self.adaptation(obs_history)
        obs_and_z = np.concatenate([obs_current, latent_z], axis=-1)
        return self.policy(obs_and_z)
