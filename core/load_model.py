import joblib 
import pickle
import os

#Dyanamic Model loader of the different kinds 
def Dyanamic_model_loader(model_path,custom_class = None):
    """This program accepts different kinds of the model
       it scans the file and then intialise the dependies 
       and return a unified file to the program"""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file or directory not found at :{model_path}")

    #Case 1:For TensorFlow saved models or huggin face transformers
    if os.paths.exists(model_path):
        contents = os.listdir(model_path)
        if 'saved_model.pb' in contents:#for the models of the tensorflow 
            import tensorflow as tf 
            raw_model = tf.keras.models.load_model(model_path)#laoding the tensorflow models
            return UnifiedPredictor(raw_model,'tensorflow')#using the UnifiedPredictor for wrapping the model for comman interface
        elif any(f in contents for f in ['config.json' , 'model.safetensors','pytorch_model.bin']): #for the of the hugging face which are already saved or pretrained model 
            from transformers import AutoModel
            raw_model = AutoModel.pretrained(model_path)
            return UnififiedPrecitor(raw_model,"huggingface")
        raise ValueError("Directory detected But unrecgnized file founded")

    #file extension - it extracts the file extension 
    _, ext = os.path.splitext(model_path)#for the direct files not the director that we have already coded 
    ext = ext.lower()

    #traditional Ml Models 
    if ext == '.joblib': #for common joblib saved model
        with open(model_path,'rb') as f:
            return UnifiedPredictor(joblib.load(model_path) , "sklearn")
    elif ext == '.pkl':#for the pickle model 
        with open (model_path,'rb') as f:
            return UnifiedPredictor(pickle.load(f),"sklearn")

    #kears or tensorflow:
    elif ext in ['.keras' , '.h5']:#keras and 
        import tensorflow as tf 
        return UnifiedPredictor(tf.keras.models.load_model(model_path) , "tensorflow")

    #for the pytorch Weights or Objects 
    elif ext in ['.pt' , '.pth' , '.ckpt']: #for pytorch weights , pytorch checkpoints , multiple framwork 
        import torch 
        if custom_class is not None:
            # Instantiate architecture , load safe weights 
            raw_model = custom_class()
            try:
                raw_model.load_state_dict(torch.load(model_path , map_location='cpu' , weights_only=True))

            except Exception:
                raw_model = torch.load(model_path , map_location= 'cpu')

            else:
                raw_model = torch.load(model_path,map_location='cpu')

            if hasattr(raw_model , 'eval'):
                raw_model.eval()

            return UnifiedPredictor(raw_model,'pytorch')


    #for the ONNX Runtimes 
    elif ext  == '.onnx':
        import onnxruntime as ort 
        raw_model = ort.InferenceSession(model_path)
        return UnifiedPredictor(raw_model , 'onnx')

    raise ValueError(f"The model format '{ext}' is completely unknown to this pipeline")
    


    

    



               
