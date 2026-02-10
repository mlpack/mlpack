## YOLOv3 and YOLOv3Tiny

The `YOLOv3` and `YOLOv3Tiny` classes implement the models from
the paper ["YOLOv3: An Incremental Improvement"](https://arxiv.org/abs/1804.02767). `YOLOv3` is a simple
object detection algorithm that takes in a image and makes
thousands of guesses in a single neural network pass, which makes
these models really easy and quick to use.

**NOTE**: At the current time, only prediction is supported by the `YOLOv3`
and `YOLOv3Tiny` classes. Support for training and fine-tuning is in progress.

#### Simple usage example:

```c++
// Download: https://models.mlpack.org/yolo/yolov3-320.bin
// Predict bounding boxes from an image using `YOLOv3`
size_t imgSize = 320;
mlpack::YOLOv3 model;
mlpack::Load("yolov3-320.bin", model);

// Download: https://github.com/pjreddie/darknet/blob/master/data/dog.jpg
arma::fmat image;
mlpack::ImageInfo info;
mlpack::Load("dog.jpg", image, info, true);

// Image pixel values must be between 0 and 1
arma::fmat preprocessedImage = image / 255.0f;

// Resize the image so that original image ratio is kept
// while making the input image square using the letterbox transform.
mlpack::ImageInfo preprocessedInfo = info;
mlpack::LetterboxImages(preprocessedImage, preprocessedInfo, imgSize, imgSize, 0.5f);

// Channels in the image must be grouped for doing convolutions.
preprocessedImage = mlpack::GroupChannels(preprocessedImage, preprocessedInfo);

// Get raw outputs from model.
arma::fmat detections;
model.Predict(preprocessedImage, detections);
```

#### Quick Links:

 * [Constructors](#constructors): create `YOLOv3` objects.
 * [`Predict()`](#predicting-bounding-boxes): predict bounding boxes in an image.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [YOLOv3Tiny](#yolov3tiny) example using instead of `YOLOv3`.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.

#### See also:
 * [YOLOv3 paper](https://arxiv.org/abs/1804.02767) where the YOLOv3 architecture is described.
 * [`DAGNetwork`](/src/mlpack/methods/ann/dag_network.hpp) is used internally to represent the model.
 * [Weights](https://models.mlpack.org/yolo/) you can download for loading YOLOv3 models and their weights.

### Constructors

Construct a `YOLOv3` object using one of the constructors below.
Defaults and types are detailed in the
[Constructor Parameters](#constructor-parameters) section below.

#### Forms

 * `model = YOLOv3(imgSize, numClasses, predictionsPerCell, anchors)`
   - Initialize the model that takes an image whose width and height are `imgSize`.
   - The model will be able to predict `numClasses` different number of classes. The pretrained weights were trained on COCO which has 80 classes.
   - A YOLOv3 model has 3 heads where each head makes `predictionsPerCell` number of predictions per grid cell.
   - The YOLOv3 model uses anchors collected from the training set using k-means. For YOLOv3 pretrained weights, the anchors are `(10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326)`, which represent (w0, h0, w1, h1 ... ). More information is described in the paper, in particular the "Bounding Box Prediction" section.

---

#### Constructor Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `imgSize` | `size_t` | Width and height of input images. |
| `numClasses` | `size_t` | Number of classes in the dataset. |
| `predictionsPerCell` | `size_t` | Number of different predictions made per grid cell in each detection head. |
| `anchors` | `std::vector<typename MatType::elem_type>` | Anchors for predicting the widths and heights of bounding boxes. |

### Predicting Bounding Boxes

Once the weights are loaded and the input images are preprocessed, you can do inference with `Predict`.

#### Forms

 * `model.Predict(input, output)`

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `input` | `MatType` | Input image. See [example](#simple-usage-example) for details on preprocessing the input image |
| `output` | `MatType` | Bounding box detections. |


The output detections is a matrix of shape `(numAttributes, numDetections)`. `numAttributes` is the number of data points that represent a bounding box, including `cx`, `cy`, `w`, `h` to represent the bounding box, the objectness score, which represents how likely an object is present within the box, and then all the class probabilities, which present the probability of a certain object given an object is in the image. You can do `objectness * class probability` to find the probability of some object being within the box.

### Other Functionality

 * `YOLOv3` can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).
 * `model.Model()` will return the underlying
   [`DAGNetwork`](/src/mlpack/methods/ann/dag_network.hpp) object that represents the model architecture.

### `YOLOv3Tiny`

Example using `YOLOv3Tiny` instead of `YOLOv3`. Since both `YOLOv3` and `YOLOv3Tiny` classes have identical APIs, simply change the weights and the class being used.


```c++
// Download: https://models.mlpack.org/yolo/yolov3-tiny-coco.bin
// Predict bounding boxes from an image using `YOLOv3Tiny`
size_t imgSize = 320;
mlpack::YOLOv3Tiny model;
mlpack::Load("yolov3-tiny-coco.bin", model);

// Download: https://github.com/pjreddie/darknet/blob/master/data/dog.jpg
arma::fmat image;
mlpack::ImageInfo info;
mlpack::Load("dog.jpg", image, info, true);

// Image pixel values must be between 0 and 1
arma::fmat preprocessedImage = image / 255.0f;

// Resize the image so that original image ratio is kept
// while making the input image square using the letterbox transform.
mlpack::ImageInfo preprocessedInfo = info;
mlpack::LetterboxImages(preprocessedImage, preprocessedInfo, imgSize, imgSize, 0.5f);

// Channels in the image must be grouped for doing convolutions.
preprocessedImage = mlpack::GroupChannels(preprocessedImage, preprocessedInfo);

// Get raw outputs from model.
arma::fmat detections;
model.Predict(preprocessedImage, detections);
```

### Advanced Functionality: Template Parameters

The `YOLOv3` class also supports several template parameters, which can be
used for custom behavior.  The full signature of the class is as follows:

```
YOLOv3<OutputLayerType
       InitializationRuleType
       MatType>
```

 * `OutputLayerType`: the loss function used to train the model. **NOTE**: this is a work in progress.
 * `InitializationRuleType`: the way that weights are initialized before
   training.
 * `MatType`: specifies the type of matrix used for learning and internal
   representation of weights and biases. The pretrained weights used float-32, so that default is `arma::fmat`.
