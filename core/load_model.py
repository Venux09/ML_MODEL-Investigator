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
        if 'saved_model.pb' in contents:
            import tensorflow as tf 
            raw_model = tf.keras.models.load_model(model_path)
            return UnifiedPredictor(raw_model,'tensorflow')
        elif any(f in contents for f in ['config.json' , 'model.safetensors','pytorch_model.bin']):
            from transformers import AutoModel
            raw_model = AutoModel.pretrained(model_path)
            return UnififiedPrecitor(raw_model,"huggingface")
        raise ValueError("Directory detected But unrecgnized file founded")

    #file extension - it extracts the file extension 
    _, ext = os.path.splitext(model_path)
    ext = ext.lower()

    #traditional Ml Models 
    if ext == '.joblib':
        with open(model_path,'rb') as f:
            return UnifiedPredictor(joblib.load(model_path) , "sklearn")
    elif ext == '.pkl':
        with open (model_path,'rb') as f:
            return UnifiedPredictor(pickle.load(f),"sklearn")

    #kears or tensorflow:
    elif ext in ['']


    


    

    



               
