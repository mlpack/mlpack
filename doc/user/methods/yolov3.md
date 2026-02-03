## YOLOv3 and YOLOv3-tiny

The `YOLOv3` and `YOLOv3-tiny` classes implement the models from
the paper "YOLOv3: An Incremental Improvement". `YOLOv3` is a simple
object detection algorithm that takes in a square image and makes
thousands of guesses in a single neural network pass, which makes
these models really easy and quick to use.

Both the YOLOv3 and YOLOv3-tiny classes have identical APIs.

**NOTE**: Currently, training is being implemented and not ready yet.

#### Simple usage example:

```cpp
  mlpack::YOLOv3<mlpack::EmptyLoss, mlpack::RandomInitialization, arma::fmat> model;
  mlpack::Load(modelFile, model);

  arma::fmat image;
  ImageInfo info;
  Load(inputFile, image, info, true);

  arma::fmat preprocessedImage = image / 255.0f;
  ImageInfo preprocessedInfo = info;
  LetterboxImages(preprocessedImage, preprocessedInfo, imgSize, imgSize, 0.5f);
  preprocessedImage = mlpack::GroupChannels(preprocessedImage, preprocessedInfo);

  arma::fmat detections;
  model.Predict(preprocessedImage, detections);
```

### Constructors

* `YOLOv3(imgSize, numClasses, predictionsPerCell, anchors)
  - `imgSize` sets the input image width and height dimensions.
  - `numClasses` sets the number of classes the model can detect.
  - `predictionsPerCell`: YOLOv3 has multiple detection heads. This parameter sets how many predictions per pixel.
  - `anchors` sets the anchor width and heights of the model.

### Inference

* `Predict(input, output)`
  - `input` is the input data. The dimensions must be (imgSize * imgSize * channels, batchSize).
  - `output` Resulting bounding boxes.
