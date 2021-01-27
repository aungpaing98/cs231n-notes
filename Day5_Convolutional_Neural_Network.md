## A bit of history

In around 1957, [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt), implemented the first perceptron machine named **Mark I Perceptron**. The machine was connected to a camera that used 20x20 cadmium photocells to produce 400-pixel images, and recognize the letter of alphabet with function:
$$f(x) = \begin{cases} 1 \quad \text{if w . x + b >0}\\ 0 \quad \text{otherwise} \end{cases}$$

With the update rule of:
$$w_i(t+1) = w_i(t) - \alpha (d_j - y_j(t))x_{j, i}$$

Seems familiar, isn't it. It is the same with the perceptron we are working with.

Which, in 1960, was modified with increasing the number of inputs and adjustable bias value by Widrow and Hoff.

And it was at 1986, Rumelhart et al, many layers perceptrons was tried with complete backpropagatoin method.

And in 2006, Hinton and Salakhutdinov, did some reinvigorated research in Deep Learning and with fine tunning with backpropagation, they achieved a great result.

Finally, at 2012, there is a breakthrough with ImageNet dataset with AlexNet (Convolutional Neural Network).

Although it is AlexNet which have a breakthrough, its base model architecture already exists since 1998, the year I was born, by the model `LeNet-5` developed by LeCun, Bottou, Bengio, Haffner. But due to the lack of calculation power and dataset, they did not get quite the result.

And the modern study of biological on human vision system state that our vision system is composed of main three types of cells.
- Simple cells : Response to light orientation.
- Complex cells : Response to light orientation and movements.
- Hypercomplex cells : response to movement with an end point.

### Fast-forward to Today: ConvNet are Everywhere

With the advanced in hardward like GPU, and the explosive of dataset from the internet, ConvNet are everywhere in our daily life. It got applied from simple tasks like Image Classification, Object Detection, Semantic Segmentation to Image Captioning. And it is also popular in the field of anomaly detection for automation tasks.

There are also many fileds in which ConvNet are involved with. Transportation, Astrophysis, Movie and multimedia and so on.

### Fully Connected Layers

Back when we do not use ConvNet, we will first need to flatten the image. Then connect each input with a weigth value. For example, if the input image have dimension of `32, 32, 3`, then after flatten it will be `32 * 32 * 3 = 3072`, and the weight will have `3072 * 10` to output a value of `10`. This is only one layer and the spatial information is not preserve in this case.
![](resources/fully_conv.gif)
While in fully connected, we will need large weigth size to cover the whole image, we can use filter in convolutional layers to cover the image too.

Also, to understand convolutional operations:
![](resources/day3.gif)

Remember, when doing a convolution operation, its dimension get reduced at each side. So, to calculate the output size of the feature map, we can use the equation of:
$$O = \frac{I + 2P}{S} + 1$$

### Pooling Layers

Pooling operation in the ConvNet have the effect of reducing the feature size of each output from the activation function, and is called the downsampling. It is a commonly used method as it can reduce the memory size and calculation size for the model. Typical pooling methods include:
- Max Pooling with (2, 2 filter and 2, 2 stride)
- Average Pooling (2, 2 filter and 2, 2 stride)
- Strided Convolution (3, 3 filter and 2, 2 stride)

[Training online and visualization of Model Architecture](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html).