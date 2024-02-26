#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME huber_regression

#include <mlpack/core/util/mlpack_main.hpp>

#include "huber_regression.hpp" // Assuming your HuberRegression implementation

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Huber L1-regularized Regression and Prediction");

// Short description.
BINDING_SHORT_DESC(
  "An implementation of Huber regression for robust fitting. "
  "Given labeled data, a model can be trained and saved for future use; or, "
  "a pre-trained model can be used to predict new points."
);

// Long description.
BINDING_LONG_DESC(
  "An implementation of Huber regression for robust fitting. This method is "
  "less sensitive to outliers compared to least-squares regression. "
  "\n\n"
  "This program allows loading a Huber regression model (via the " +
  PRINT_PARAM_STRING("input_model") + " parameter) or training a Huber "
  "regression model given training data (specified with the " +
  PRINT_PARAM_STRING("training") + " parameter), or both those things "
  "at once. Additionally, this program allows prediction on a test "
  "dataset (specified with the " + PRINT_PARAM_STRING("test") + " parameter) "
  "and the prediction results may be saved with the " +
  PRINT_PARAM_STRING("predictions") + " output parameter. The trained "
  "Huber regression model may be saved using the " +
  PRINT_PARAM_STRING("output_model") + " output parameter."
  "\n\n"
  "The training data, if specified, may have class labels as its last "
  "dimension. Alternatively, the " + PRINT_PARAM_STRING("labels") + " "
  "parameter may be used to specify a separate matrix of labels."
  "\n\n"
  "When a model is being trained, there are several options. There are "
  "various parameters for the optimizer; the " +
  PRINT_PARAM_STRING("max_iterations") + " parameter specifies the maximum "
  "allowed iterations, and the " + PRINT_PARAM_STRING("tolerance") + " "
  "parameter specifies the tolerance for convergence. For more details on "
  "specific parameters, refer to the C++ interface documentation."
  "\n\n"
  "This implementation of Huber regression does not support multi-class "
  "problems. Labels must be numerical values."
);

// Example.
BINDING_EXAMPLE(
  "As an example, to train a Huber regression model on the data '" +
  PRINT_DATASET("data") + "' with labels '" + PRINT_DATASET("labels") + "', "
  "saving the model to '" + PRINT_MODEL("hr_model") + "', the following "
  "command may be used:"
  "\n\n" +
  PRINT_CALL("huber_regression", "training", "data", "labels", "labels",
            "max_iterations", 1000, "tolerance", 1e-6, "output_model", "hr_model",
            "print_training_accuracy", true) +
  "\n\n"
  "Then, to use that model to predict values for the dataset '" +
  PRINT_DATASET("test") + "', storing the output predictions in '" +
  PRINT_DATASET("predictions") + "', the following command may be used: "
  "\n\n" +
  PRINT_CALL("huber_regression", "input_model", "hr_model", "test", "test",
            "predictions", "predictions"));

// See also...
BINDING_SEE_ALSO("@linear_regression", "#linear_regression");
BINDING_SEE_ALSO("@ridge_regression", "#ridge_regression");
BINDING_SEE_ALSO("Huber regression on Wikipedia",
                "https://en.wikipedia.org/wiki/Huber_loss_function");
BINDING_SEE_ALSO(":HuberRegression C++ class documentation",
                "@src/mlpack/methods/huber_regression/huber_regression.hpp");

//
