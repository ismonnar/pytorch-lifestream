import os
import torch
import torch.nn as nn
import onnxruntime as ort

class ONNXInferenceModel(nn.Module):
    def __init__(self, path, model=None, dl=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = path
        self.providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
        
        if model is not None:
            batch = next(iter(dl))
            features, names, seq_len = self.preprocessing(batch)
            model._col_names = names
            model._seq_len = seq_len
            model.example_input_array = features
            self.export(path, model)
            model = path
        
        self.ort_session = ort.InferenceSession(
            model,
            providers=self.providers
        )

    def preprocessing(self, batch):
        features = torch.stack([v for v in batch[0].values()])
        names = [k for k in batch[0].keys()]
        seq_len = batch[1]
        return features, names, seq_len

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
        inputs, _, _ = self.preprocessing(x)
        return self.ort_session.run(None, {"input": inputs.numpy()})

    def to(self, device):
        # onnx runtime chooses it's own way
        return self

    def size(self):
        return os.path.getsize(self.model_name)
    
    def predict(self, dl):
        pred = torch.Tensor()
        for batch in dl:
            out = torch.Tensor(self(batch)).squeeze(dim=0)
            pred = torch.cat((pred, out))
        return pred