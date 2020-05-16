import tensorflow as tf
import os.path as path 

# folder = 'acd'
# model_names = ['dnn', 'cnn', 'lstm']

folder = 'subjectivity'
model_names = ['cnn_2']

for i in range(0, len(model_names)):
    model_name = model_names[si]
    img_location = path.join('model_png', folder + '_' + model_name + '.png')
    model = tf.keras.models.load_model(path.join(folder, path.join('tf_models', model_name + '_model')))

    tf.keras.utils.plot_model(model, to_file=img_location, show_shapes=True)
