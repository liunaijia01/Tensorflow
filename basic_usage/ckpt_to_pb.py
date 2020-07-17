import tensorflow as tf
import sys
checkpoint_dir = ""
trained_checkpoint_prefix = checkpoint_dir
export_dir = "saved_model_test"
graph = tf.Graph()
# config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()