import os 
import numpy as np 


class UnifiedPredictor:
     """
     this function can wrap any ML / Dl models for a single , clean 
     interface for the AI evaluation"""

     def __init__(self , raw_model , framework_type):
          self.model = raw_model
          self.framwork_type = framework_type

     def predict(self , X):
          """
          accepts any standart numpy array  or pandas Dataframes and routes it correctly."""

          #for numpy array data 
          if hasattr(X,"to_numpy"):
               X_np = np.to_numpy()

          else:
               X_np = np.array(X)

          try:

              if self.framwork_type == 'sklearn':
                   return self.model.predict(X_np)

              elif self.framwork_type == 'pytorch':
                   import torch 
                   #converting the numpy array to PyTorch Tensor
                   tensor_X = torch.tensor(X_np,dtype=torch.float32)
                   with torch.no_grad():
                        outputs = self.model(tensor_X)
                        return outputs.cpu.numpy()

              elif self.framwork_type == 'tensorflow':
                   #keras 
                   return self.model.predict(X_np,verbose = 0)

              elif self.framwork_type == 'onnx':
                   #ONNX dictionary mapping to the tensor 
                   input_name = self.model.get_inputs()[0].name
                   return self.model.run(None,{input_name :X_np.astype(np.float32)})[0]
              