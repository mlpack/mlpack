## YOLOv3 and YOLOv3Tiny

The `YOLOv3` and `YOLOv3Tiny` classes implement the models from
the paper ["YOLOv3: An Incremental Improvement"](https://arxiv.org/abs/1804.02767). `YOLOv3` is a simple
object detection algorithm that takes in an image and predicts multiple bounding
boxes in a single forward pass of the neural network.

<p align="center">
  <img src="../../img/dog.jpg" alt="dog, bicycle and truck">
</p>

```
@article{DBLP:journals/corr/abs-1804-02767,
  author       = {Joseph Redmon and
                  Ali Farhadi},
  title        = {YOLOv3: An Incremental Improvement},
  journal      = {CoRR},
  volume       = {abs/1804.02767},
  year         = {2018},
  url          = {http://arxiv.org/abs/1804.02767},
  eprinttype    = {arXiv},
  eprint       = {1804.02767},
  timestamp    = {Mon, 13 Aug 2018 16:48:24 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1804-02767.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

**NOTE**: At the current time, only prediction is supported by the `YOLOv3`
and `YOLOv3Tiny` classes. Support for training and fine-tuning is in progress.

#### Simple usage example:

```c++
// Download: https://models.mlpack.org/yolo/yolov3-320-coco.bin
mlpack::YOLOv3 model;
mlpack::Load("yolov3-320-coco.bin", model);

// Download: https://github.com/pjreddie/darknet/blob/master/data/dog.jpg
arma::fmat image;
mlpack::ImageInfo info;
mlpack::Load("dog.jpg", image, info, true);

// Predict bounding boxes from an image using `YOLOv3` and draw them.
// It will ignore boxes when the model's confidence of the object is below
// `ignoreThreshold`
const double ignoreThreshold = 0.7;
model.Predict(image, info, ignoreThreshold);

// Save to "output.jpg"
mlpack::Save("output.jpg", image, info, true);
```

<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick Links:

 * [Constructors](#constructors): create `YOLOv3` objects.
 * [`Predict()`](#predicting-bounding-boxes): predict bounding boxes in an
   image.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting the model.
 * [`YOLOv3Tiny` class](#yolov3tiny) for efficient low-resource detection.
 * [Pretrained weights](#pretrained-weights) for different YOLO models.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.

#### See also:
 * [Object Detection on Wikipedia](https://en.wikipedia.org/wiki/Object_detection)
 * [You Only Look Once (original YOLO paper)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
 * [YOLOv3 paper](https://arxiv.org/abs/1804.02767) where the YOLOv3 architecture is described.
 * [`DAGNetwork`](/src/mlpack/methods/ann/dag_network.hpp) is used internally to represent the model.

### Constructors

Construct a `YOLOv3` object using one of the constructors below.
Defaults and types are detailed in the
[Constructor Parameters](#constructor-parameters) section below.

 * `model = YOLOv3()`
   - Create an uninitialized YOLOv3 model.
   - After this step, load [pretrained weights](#pretrained-weights) into the
     model with [`Load()`](../load_save.md#load).

---

### Predicting Bounding Boxes

Once the weights are loaded, you can compute likely object bounding boxes with with `Predict()`.

#### Forms

<!-- TODO: could have one that takes input and output file names? so image Load/Save is abstracted too. -->

 * `model.Predict(image, opt, ignoreThresh)`
   - Predict objects in the given `image` (with metadata `opts`) and
     draw the detections onto that image.
   - Bounding boxes will not be drawn unless their confidence is
     greater than `ignoreThreshold`.


| **name** | **type** | **description** | **default**|
|----------|----------|-----------------|------------|
| `image` | `arma::fmat` | Input image. See [example](#simple-usage-example) for details on loading an image with mlpack | n/a |
| `opt` | `ImageOptions` | Image metadata. | n/a |
| `ignoreThreshold` | `double` | Minimum confidence to have the corresponding bounding box drawn onto `image`. | `0.7` |

---

 * `model.Predict(input, output)`
  - Takes in a preprocessed `input`. See [example](#simple-examples)
  - `output` stores the raw detection data. The shape of the output matrix will be `(numAttributes * numDetections, batchSize)`.
  - You can get the `numAttributes` of the model from [`model.NumAttributes()`](#other-functionality).
  - Each bounding box is made up of `numAttributes` number of data points. This includes `cx`, `cy`, `w`, `h`, objectness and class probabilities.
  - Objectness means how likely an object is in the given box.
  - The class probability means that given there's an object in the box, what's the probability that it's this class.


| **name** | **type** | **description** |
|----------|----------|-----------------|
| `input` | `MatType` | Input image. See [example](#simple-usage-example) for details on preprocessing the input image |
| `output` | `MatType` | Raw outputs of the model |

---

### Other Functionality

 * `YOLOv3` can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).

 * `Model()` will return the underlying
   [`DAGNetwork`](/src/mlpack/methods/ann/dag_network.hpp) object that represents the model architecture.

 * `ImageSize()` will return the width and height of the image after
    preprocessing. The `yolov3-320.bin` model for example converts the
    input image to shape `(320, 320, 3)`.

 * `NumAttributes()` will return the number of attributes that make up
    a bounding box prediction. For example, if the model predicts 80
    classes, it will return 85 to include the bounding box coordinates
    and its objectness score.

 * `ClassNames()` will return a vector of strings, each being the name
    of a class the model can predict.

### `YOLOv3Tiny`

<!-- TODO: update when YOLOv3Tiny is updated. -->

### Pretrained weights

Because training a `YOLOv3` model from scratch is time-consuming,
a number of pretrained models are available for download:

The format for the name of each pretrained model is `<model name>-<image size>-<finetuned dataset name>.bin`.

 * [`https://models.mlpack.org/yolo/yolov3-320-coco.bin`](https://models.mlpack.org/yolov3/yolov3-320-coco.bin)

<!-- TODO: update and add other models -->

### Simple Examples

<!-- TODO: add example using raw output -->

### Advanced Functionality: Template Parameters

The `YOLOv3` class also supports several template parameters, which can be
used for custom behavior.  The full signature of the class is:

```
YOLOv3<MatType
       OutputLayerType
       InitializationRuleType>
```

 * `MatType`: specifies the type of matrix used for learning and internal
   representation of weights and biases. The pretrained weights used float-32, so that default is `arma::fmat`.
 * `OutputLayerType`: the loss function used to train the model. **NOTE**:
   loss functions for object detection models are a work in progress.
 * `InitializationRuleType`: the way that weights are initialized before
   training.
