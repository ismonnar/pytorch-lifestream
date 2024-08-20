import os
import numpy as np
import psutil
import torch
from tqdm import tqdm
import pytorch_lightning as pl
import onnxruntime as ort

class ONNXInferenceModel(pl.LightningModule):
    def __init__(self, model_path, model=None, dl=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_path
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.so = ort.SessionOptions()
        self.so.log_severity_level = 0
        self.so.intra_op_num_threads = psutil.cpu_count(logical=True)
        self.so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if model is not None:
            batch = next(iter(dl))
            features, names, seq_len = self.preprocessing(batch)
            model._col_names = names
            model._seq_len = seq_len
            model.example_input_array = features
            self.export(model_path, model)
            model = model_path
        
        self.ort_session = ort.InferenceSession(
            model,
            sess_options=self.so,
            providers=self.providers
        )
        self.binding = self.ort_session.io_binding()

    def stack(self, x):
        return torch.stack([v for v in x[0].values()])

    def preprocessing(self, x):
        features = self.stack(x)
        names = [k for k in x[0].keys()]
        seq_len = x[1]
        return features, names, seq_len
    
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def export(self,
                path: str,
                model
            ) -> None:
        
        model.to_onnx(path,
                    export_params=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                                "input": {
                                    0: "features",
                                    1: "batch_size",
                                    2: "seq_len"
                                },
                                "output": {
                                    0: "batch_size",
                                    1: "seq_len"
                                }
                            }
                    )
    
    def forward(self, x):
        inputs = self.to_numpy(self.stack(x))
        output = self.ort_session.run(None, {"input": inputs})
        return torch.tensor(output[0], dtype=torch.float32)

    def to(self, device="cpu", device_id="auto"):
        self._device_type = str(device)
        self._device_id = str(device_id-1) if isinstance(device_id, int) else device_id
        return self

    def size(self):
        return os.path.getsize(self.model_name)
    
    def predict(self, dl):
        pred = list()
        desc = 'Predicting DataLoader:'
        with torch.no_grad():
            for batch in tqdm(dl, desc=desc):
                output = self(batch)
                pred.append(output)
        return pred