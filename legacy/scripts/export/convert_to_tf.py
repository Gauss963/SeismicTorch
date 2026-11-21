from onnx_tf.backend import prepare
import onnx

TF_PATH = './model/tf_finetuned' # where the representation of tensorflow model will be stored
ONNX_PATH = './model/ONNX_finetuned.onnx' # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model
tf_rep = prepare(onnx_model)  # creating TensorflowRep object
tf_rep.export_graph(TF_PATH)