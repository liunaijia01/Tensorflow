import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def convert_h5_to_pb(h5_path, log_dir, pb_name):
    model = tf.keras.models.load_model(h5_path, compile=False)
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=log_dir,
                      name=pb_name,
                      as_text=False)
