import tensorflow as tf
import numpy as np
from PIL import Image
from time import time
from PySide6.QtCore import QObject, Signal, QByteArray, QCoreApplication, QThread
import cv2

class StyleTransfer(QObject):
    #Description: This class contains an implementation of the Neural Style Transfer algorithm
    #             using the pretrained VGG19 model. The algorithm is based on the paper:
    #             "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
    #             https://arxiv.org/abs/1508.06576
    #             The implementation is based on the tutorials:
    #             https://www.tensorflow.org/tutorials/generative/style_transfer
    #             https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    #             and a previous implementation of the algorithm in MATLAB by the author of this code, 
    #             based upon th
    #
    #
    #
    # Signals to update the image in the GUI
    result = Signal(np.ndarray)
    pb_progress = Signal(int)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.output_image = None
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1',
                                'block2_conv1',
                                'block3_conv1',
                                'block4_conv1',
                                'block5_conv1']
        self.total_variation_weight = 45
        self.epochs = 10
        
    def preprocess_image_neural(self, image):
        # Preprocess image for VGG19
        image = tf.keras.applications.vgg19.preprocess_input(image)
        image = image.copy()
        image = tf.image.resize(image, (224, 224))
        image = image[tf.newaxis, :]
        return image
    
    def deprocess_image_neural(self, processed_img):
        # Deprocess image for presentation
        image = processed_img.copy()
        image = np.array(image).reshape((224, 224, 3))
        # Add mean values of VGG19 input layer
        image[:, :, 0] += 103.939
        image[:, :, 1] += 116.779
        image[:, :, 2] += 123.68
        image = image[:, :, ::-1]
        # Equalize histogram to desaturate pure colors
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.9
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        image = np.clip(image, 0, 255).astype('uint8')


        return image


    def vgg_layers(self, layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    class StyleContentModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers):
            super(StyleTransfer.StyleContentModel, self).__init__()
            self.vgg = self.vgg_layers(style_layers + content_layers)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False

        def vgg_layers(self, layer_names):
            vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
            vgg.trainable = False

            outputs = [vgg.get_layer(name).output for name in layer_names]
            model = tf.keras.Model([vgg.input], outputs)
            return model

        def call(self, inputs):
            inputs = inputs * 255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                              outputs[self.num_style_layers:])

            style_outputs = [StyleTransfer.gram_matrix(self, style_output)
                             for style_output in style_outputs]

            content_dict = {content_name: value
                            for content_name, value
                            in zip(self.content_layers, content_outputs)}

            style_dict = {style_name: value
                          for style_name, value
                          in zip(self.style_layers, style_outputs)}

            return {'content': content_dict, 'style': style_dict}

    #def transfer_style(self, content_image, style_image):
    
    def run(self, content_image, style_image):
    
        epochs = self.epochs
        steps_per_epoch = 200
        
        content_layers = self.content_layers
        style_layers = self.style_layers

        content_image = self.preprocess_image_neural(content_image)
        style_image = self.preprocess_image_neural(style_image)

        extractor = self.StyleContentModel(style_layers, content_layers)

        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        opt = tf.optimizers.legacy.Adam(learning_rate=0.05, beta_1=0.99, epsilon=1e-1)

        style_weight = 1e-1
        content_weight = 1e4

        def compute_loss(outputs):
            style_outputs = outputs['style']
            content_outputs = outputs['content']

            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) 
                                   for name in style_outputs.keys()])
            style_loss *= style_weight / len(style_layers)

            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) 
                                     for name in content_outputs.keys()])
            content_loss *= content_weight / len(content_layers)
            loss = style_loss + content_loss
            return loss, style_loss, content_loss

        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss, style_loss, content_loss = compute_loss(outputs)
                loss = loss + self.total_variation_weight * tf.image.total_variation(image)

                print("Style loss: {}".format(style_loss), "Content loss: {}".format(content_loss), end="\r")

            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])

        # Training loop
        image = tf.Variable(content_image)

        for n in range(epochs):
            for m in range(steps_per_epoch):
                train_step(image)
                QCoreApplication.processEvents()
            self.output_image = self.deprocess_image_neural(image.numpy())
            byte_array = QByteArray(self.output_image.tobytes())
            self.result.emit(byte_array)
            self.pb_progress.emit(n)
        self.finished.emit()

        

