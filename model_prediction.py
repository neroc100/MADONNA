import tensorflow as tf
import numpy as np

# add the actual model here 
# from the paper we know: 
# [x] shallow neural network 
# [ ] with float16 quantization (from Tensorflow optimisation toolkit)
# [x] uses 13 identified important features
# [x] 28 nodes in the hidden layer, [x] ReLU activation 
# [ ] train for 50 epochs 
# [ ] train with a batch size of 128
# [ ] Adam optimizer with a learning rate of 0.001 (GD; has momentum)
# [ ] early stopping to prevent overfitting
# [x] softmax for the classification layer

######################### SNN from Hands on Deep Learning ###########################
class ShallowNeuralNet(nn.Module):
    def __init__(self, input_width: int, hidden_layer_width: int, output_width):
        super().__init__()
        self.hidden_layer = nn.Linear(input_width, hidden_layer_width)
        self.hidden_relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_layer_width, output_width)
        self.output_activation = torch.nn.Softmax(dim=0)

    def forward(self, input):
        hidden_trainable_output = self.hidden_layer(input)
        hidden_relu_output = self.hidden_relu(hidden_trainable_output)
        output = self.output_layer(hidden_relu_output)
        softmax_output = self.output_activation(output)
        return softmax_output

#################### SNN Instance creation from Hands on Deep Learning ##############
shallow_nn_instance = ShallowNeuralNet(13, 28, 2)
shallow_nn_instance.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

example_input = torch.ones(13)
shallow_nn_instance(example_input)

########################## Loading OG paper model ###################################

interpreter = tf.lite.Interpreter(model_path='lite_model_optimized_float16.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

input_details[0]['shape']

input_shape = input_details[0]['shape']
X_test = np.array([15,2,4,365,5,8,0,1,3.189898095464287,0,1.0,3,31], dtype=np.uint32)

inp = np.expand_dims(X_test, axis=0)

inp = inp.astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], inp)

interpreter.invoke()

pred = interpreter.get_tensor(output_details[0]['index'])[0][0]


print(pred)