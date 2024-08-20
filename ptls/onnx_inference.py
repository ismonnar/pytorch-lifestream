import os
import torch
from tqdm import tqdm
import pytorch_lightning as pl
import onnxruntime as ort

class ONNXInferenceModel(pl.LightningModule):
    def __init__(self, model_path, model=None, dl=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_path
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
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
            providers=self.providers
        )

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
    
    def forward(self, x, dtype: torch.dtype = torch.float16):
        inputs = self.to_numpy(self.stack(x))
        output = self.ort_session.run(None, {"input": inputs})
        return torch.tensor(output[0], dtype=dtype)

    def to(self, device):
        return self

    def size(self):
        return os.path.getsize(self.model_name)
    
    def predict(self, dl, dtype: torch.dtype = torch.float32):
        pred = list()
        desc = 'Predicting DataLoader'
        with torch.no_grad():
            for batch in tqdm(dl, desc=desc):
                output = self(batch, dtype=dtype)
                pred.append(output)
        return pred