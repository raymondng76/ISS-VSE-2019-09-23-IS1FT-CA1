import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util
from keras import backend as K
import keras

def convert_keras_to_tensorflow(keras_model, output_model):
	# Disable learning phase on Keras backend before loading weights
	K.set_learning_phase(0)
	model = keras.models.load_model(keras_model)
	output_node_name = [node.op.name for node in model.outputs]
	
	# Get keras session and convert to a graph
	sess = K.get_session()
	constant_graph = graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            output_node_name)
	graph_io.write_graph(constant_graph, '', output_model, as_text=False)