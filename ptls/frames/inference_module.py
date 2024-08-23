import os
import pandas as pd
import pytorch_lightning as pl
import torch
import onnxruntime as ort

from tqdm import tqdm
from ptls.data_load.padded_batch import PaddedBatch


class InferenceModule(pl.LightningModule):
    def __init__(self, model, model_out_name='out', pandas_output=False):
        super().__init__()

        self.model = model
        self.model_out_name = model_out_name
        self.pandas_output = pandas_output

    def forward(self, x: PaddedBatch):
        out = self.model(x)
        if self.pandas_output:
            return self.to_pandas(out)
        return out

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def to_pandas(self, x):
        tensors = []
        for tensor in x:
            tensor = self.to_numpy(tensor)
            tensors.append(tensor)
        return pd.DataFrame({f'{self.model_out_name}': tensors})


class InferenceModuleMultimodal(InferenceModule):
    def __init__(self, model, pandas_output=False, model_out_name='out', col_id = 'epk_id'):
        super().__init__(model)

        self.model = model
        self.pandas_output = pandas_output
        self.model_out_name = model_out_name
        self.col_id = col_id

    def forward(self, x: PaddedBatch):
        x, batch_ids = x
        out = self.model(x)
        x_out = {self.col_id : batch_ids, self.model_out_name: out}
        if self.pandas_output:
            return self.to_pandas(x_out)
        return x_out

    def to_pandas(self, x):
        expand_cols = []
        scalar_features = {}

        for k, v in x.items():
            if type(v) is torch.Tensor:
                v = self.to_numpy(v)
            if type(v) is list or len(v.shape) == 1:
                scalar_features[k] = v
            elif len(v.shape) == 2:
                expand_cols.append(k)
            else:
                scalar_features[k] = None

        dataframes = [pd.DataFrame(scalar_features)]
        for col in expand_cols:
            v = x[col].detach().cpu().numpy()
            dataframes.append(pd.DataFrame(v, columns=[f'{col}_{i:04d}' for i in range(v.shape[1])]))

        return pd.concat(dataframes, axis=1)

class ONNXInferenceModule(InferenceModule):
    def __init__(self, model, dl, model_out_name='emb.onnx', pandas_output=False):
        super().__init__(model)
        self.model = model
        self.pandas_output = pandas_output
        self.model_out_name = model_out_name
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        batch = next(iter(dl))
        features, names, seq_len = self.preprocessing(batch)
        model._col_names = names
        model._seq_len = seq_len
        model.example_input_array = features
        self.export(self.model_out_name, model)
    
        self.ort_session = ort.InferenceSession(
            self.model_out_name,
            providers=self.providers
        )

    def stack(self, x):
        return torch.stack([v for v in x[0].values()])

    def preprocessing(self, x):
        features = self.stack(x)
        names = [k for k in x[0].keys()]
        seq_len = x[1]
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
                                    1: "hidden_size"
                                }
                            }
                    )
    
    def forward(self, x, dtype: torch.dtype = torch.float16):
        inputs = self.to_numpy(self.stack(x))
        out = self.ort_session.run(None, {"input": inputs})
        out = torch.tensor(out[0], dtype=dtype)
        if self.pandas_output:
            return self.to_pandas(out)
        return out

    def to(self, device):
        return self

    def size(self):
        return os.path.getsize(self.model_name)
    
    def predict(self, dl, dtype: torch.dtype = torch.float16):
        pred = list()
        desc = 'Predicting DataLoader'
        with torch.no_grad():
            for batch in tqdm(dl, desc=desc):
                output = self(batch, dtype=dtype)
                pred.append(output)
        return pred