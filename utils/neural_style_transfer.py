import tensorflow as tf
import numpy as np
from time import time
from PySide6.QtCore import QObject, Signal, QByteArray, QCoreApplication, QThread
import cv2

class NeuralStyleTransfer(QObject):
    
    """ 
    This class contains an implementation of the Neural Style Transfer algorithm
    using the pretrained VGG19 model. The algorithm is based on the paper:
    "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
    https://arxiv.org/abs/1508.06576

    The implementation is based on the tutorials:
    https://www.tensorflow.org/tutorials/generative/style_transfer
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    and a previous implementation of the algorithm in MATLAB by the author of this code.

    Methods
    -------
    preprocess_image_neural(image)
        Preprocess image for VGG19, resizing it to (224,224,3).
    deprocess_image_neural(processed_img, content_image_aspect_ratio)
        Deprocess VGG19 output tensor to image ndarray, reshaping 
        the result to the original aspect ratio of the content image.
    gram_matrix(input_tensor)
        Compute the Gram matrix of the input tensor.
    vgg_layers(layer_names)
        Create a model from VGG19 with specified VGG19 layer_names.
    style_transfer_model(style_layers, content_layers)
        Create the style transfer model from VGG19, with specified style and content layers,
        using vgg_layers and calculating the Gram matrix for the style output
    compute_loss(outputs)
        Compute the total loss, style loss, and content loss.
    run(content_image, style_image, epochs = 10, steps_per_epoch = 400)
        Run the Neural Style Transfer algorithm.
    """

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
        self.style_weight = 1e3
        self.content_weight = 1
        self.style_targets = None
        self.content_targets = None
        
    def preprocess_image_neural(self, image):
        """
        Preprocess image for VGG19, resizing it to (224,224,3).

        Parameters
        ----------
        image (numpy.ndarray)
            Input image array.

        Returns
        ------
        tensorflow.Tensor
            Preprocessed image tensor for VGG19 with shape (244,244,3).
        """
        # Preprocess image for VGG19
        image = tf.keras.applications.vgg19.preprocess_input(image)
        image = image.copy()
        image = tf.image.resize(image, (224, 224))
        image = image[tf.newaxis, :]
        return image
    
    def deprocess_image_neural(self, processed_img, content_image_aspect_ratio):
        """
        Deprocess image for presentation, reshaping the result to the original aspect ratio of the content image.
 
        Parameters
        ----------
        processed_img (numpy.ndarray)
            Processed image.

        content_image_aspect_ratio (float)
            Aspect ratio of the content image.

        Returns
        ------
            numpy.ndarray: Deprocessed image for presentation as uint8 ndarray.
        """
        # Deprocess image for presentation
        image = processed_img.copy()
        if content_image_aspect_ratio>1:
            image = np.array(image).reshape((224, int(224*content_image_aspect_ratio), 3))
        else:
            image = np.array(image).reshape((224*int(content_image_aspect_ratio), 224, 3))
        # Add mean values of VGG19 input layer
        image[:, :, 0] += 103.939
        image[:, :, 1] += 116.779
        image[:, :, 2] += 123.68
        image = image[:, :, ::-1]
        # Equalize histogram to desaturate pure colors
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        image = np.clip(image, 0, 255).astype('uint8')
        return image

    def gram_matrix(self, input_tensor):
        """
        Calculate the Gram matrix of the input tensor.

        Parameters
        ----------
        input_tensor (tensorflow.Tensor)
            Input tensor.

        Returns
        ------
        tensorflow.Tensor
            Gram matrix of the input sensor.
        """
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    def vgg_layers(self, layer_names):
        """
        Create a model from VGG19 with specified VGG19 layer_names.

        Parameters
        ----------
        layer_names (list)
            List of layer names.

        Returns
        ------
        tensorflow.keras.Model
            VGG model with specified layers.
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model
    
    def style_transfer_model(self, style_layers, content_layers):
        """
        Create the style transfer model from VGG19, with specified style and content layers,
        using vgg_layers(layer_names) and applying gram_matrix(input_tensor) to the style layers.

        Parameters
        ----------
        style_layers (list)
            List of style layers.
        content_layers (list)   
            List of content layers.

        Returns
        ------
        function
            Style transfer model.
        """
        vgg = self.vgg_layers(style_layers + content_layers)
        num_style_layers = len(style_layers)
        vgg.trainable = False

        def model(inputs):
            inputs = inputs * 255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:num_style_layers], outputs[num_style_layers:])

            style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]

            content_dict = {content_name: value
                            for content_name, value
                            in zip(content_layers, content_outputs)}

            style_dict = {style_name: value
                            for style_name, value
                            in zip(style_layers, style_outputs)}

            return {'content': content_dict, 'style': style_dict}

        return model
    
    def compute_loss(self, outputs):
        """
        Compute the total loss, style loss, and content loss.

        Parameters
        ----------
        outputs (dict)
            Dictionary containing style and content outputs.

        Returns
        ------
        tuple  
            Total loss, style loss, and content loss.
        """
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2) 
                                for name in style_outputs.keys()])
        style_loss *= self.style_weight / len(self.style_layers)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2) 
                                    for name in content_outputs.keys()])
        content_loss *= self.content_weight / len(self.content_layers)
        loss = style_loss + content_loss
        return loss, style_loss, content_loss

    def run(self, content_image, style_image, epochs = 10, steps_per_epoch = 400):
        """
        Run the Neural Style Transfer algorithm.

        Parameters
        ----------
        content_image (numpy.ndarray)
            Content image.
        
        style_image (numpy.ndarray)
            Style image.

        epochs (int)
            Number of training epochs.
            
        steps_per_epoch
            Number of steps per epoch.
        
        Returns
        -------
        None

        
        """

        content_layers = self.content_layers
        style_layers = self.style_layers

        content_image = self.preprocess_image_neural(content_image)
        style_image = self.preprocess_image_neural(style_image)

        self.extractor = self.style_transfer_model(style_layers,content_layers)

        self.style_targets = self.extractor(style_image)['style']
        self.content_targets = self.extractor(content_image)['content']

        opt = tf.optimizers.legacy.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = self.extractor(image)
                loss = self.compute_loss(outputs)
                # Apply total variation loss
                loss += self.total_variation_weight * tf.image.total_variation(image)

                grad = tape.gradient(loss, image)
                opt.apply_gradients([(grad, image)])

        # Training loop
        image = tf.Variable(content_image)
        
        # Get aspect ratio of content image
        aspect_ratio = content_image.shape[2]/content_image.shape[1]

        noise = tf.Variable(tf.random.uniform(content_image.shape, 0, 68))
        # mean between image and noise to start with a random image
        image = tf.Variable((0.8*image + noise) / 2)

        for n in range(epochs):
            for m in range(steps_per_epoch):
                train_step(image)
                QCoreApplication.processEvents()
            self.output_image = self.deprocess_image_neural(image.numpy(), aspect_ratio)

            byte_array = QByteArray(self.output_image.tobytes())
            self.result.emit(byte_array)
            self.pb_progress.emit(n)
        self.finished.emit()

        

