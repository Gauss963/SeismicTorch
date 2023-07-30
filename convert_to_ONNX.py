import torch
import torch.nn as nn
import torch.onnx
from def_model import EarthquakeCNN

#Function to Convert to ONNX
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    dummy_input = torch.randn(1, 1, 80, 3, requires_grad = True)

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         './model/ONNX_finetuned.onnx',       # where to save the model  
         export_params = True,  # store the trained parameter weights inside the model file 
         opset_version = 10,    # the ONNX version to export the model to 
         do_constant_folding = True,  # whether to execute constant folding for optimization 

         # input_names = ['Seismic Wave'],   # the model's input names 
         input_names = ['modelInput'],

         # output_names = ['Classification'], # the model's output names 
         output_names = ['modelOutput'],

         dynamic_axes = {'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')


if __name__ == "__main__": 
    model = torch.load('./model/EarthquakeCNN_finetuned.pth')
    Convert_ONNX()