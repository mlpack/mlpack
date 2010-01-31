#include "fastlib/fastlib.h"
#include "linear_regression_model_dev.h"
#include <deque>

int main(int argc, char *argv[]) {

  // Initialize FastExec...
  fx_init(argc, argv, NULL);

  // Initialize the random number.
  srand(time(NULL));

  // Generate a random data matrix. Each column is a point.
  int random_num_points = math::RandInt(10, 100);
  int random_num_dimensions = math::RandInt(5, 20);
  Matrix random_data_matrix;
  random_data_matrix.Init(random_num_dimensions, random_num_points);
  for (int i = 0; i < random_num_points; i++) {
    for (int j = 0; j < random_num_dimensions - 1; j++) {
      random_data_matrix.set(j, i, math::Random(-5.0, 5.0));
    }

    // The last index is the dimension for which you want to predict
    // against.
    random_data_matrix.set(random_num_dimensions - 1, i,
                           math::Random(0.0, 10.0));
  }
  printf("Generated a random matrix containing %d points of %d attributes.\n", 
	 random_num_points, random_num_dimensions);

  // Initialize the model.
  std::deque<int> initial_active_column_indices;
  for (int i = 0; i < random_num_dimensions - 1; i++) {
    initial_active_column_indices.push_back(i);
  }
  
  printf("Initializing the model.\n");
  LinearRegressionModel model;
  model.Init(random_data_matrix, initial_active_column_indices,
             random_num_dimensions - 1, true, 0.9);
  printf("Initialized the model.\n");

  for (int i = 0; i < random_num_dimensions - 1; i++) {

    printf("Computing the VIF for the feature: %d\n", i);
    // Set the right hand side to be the i-th attribute.
    LinearRegressionResult result;
    model.set_active_right_hand_side_column_index(i);
    model.Solve();
    model.Predict(random_data_matrix, &result);

    double vif = model.VarianceInflationFactor(result);
    printf("VIF for the feature: %d is %g\n", i, vif);

    // Put the i-th attribute back into the left hand side list.
    model.MakeColumnActive(i);
  }

  fx_done(fx_root);
  return 0;
}
