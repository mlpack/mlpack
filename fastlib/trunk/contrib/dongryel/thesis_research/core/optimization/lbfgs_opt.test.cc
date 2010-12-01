/** @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *
 *  @brief The test driver for the L-BFGS optimizer on some popular
 *  test functions.
 *
 *  @file lbfgs_opt_test.cc
 */

// for BOOST testing

#include "core/math/math_lib.h"
#include "core/optimization/lbfgs_dev.h"
#include <stdexcept>

namespace core {
namespace optimization {
namespace lbfgs_test {

class ExtendedRosenbrockFunction {

  private:

    int num_dimensions_;

  public:
    double Evaluate(const core::table::DensePoint &x) {
      double fval = 0;
      for(int i = 0; i < num_dimensions() - 1; i++) {
        fval = fval + 100 * core::math::Sqr(x[i] * x[i] - x[i + 1]) +
               core::math::Sqr(x[i] - 1);
      }
      return fval;
    }

    void Gradient(const core::table::DensePoint &x,
                  core::table::DensePoint *gradient) {

      gradient->SetZero();
      for(int k = 0; k < num_dimensions() - 1; k++) {
        (*gradient)[k] = 400 * x[k] * (x[k] * x[k] - x[k+1]) + 2 * (x[k] - 1);
        if(k > 0) {
          (*gradient)[k] = (*gradient)[k] + 200 * (x[k] - x[k - 1] * x[k - 1]);
        }
      }
      (*gradient)[num_dimensions() - 1] =
        200 * (x[num_dimensions() - 1] -
               core::math::Sqr(x[num_dimensions() - 2]));
    }

    int num_dimensions() const {
      return num_dimensions_;
    }

    void InitStartingIterate(core::table::DensePoint *iterate) {
      num_dimensions_ = 4 * core::math::RandInt(2, 200);
      iterate->Init(num_dimensions_);
      for(int i = 0; i < num_dimensions_; i++) {
        if(i % 2 == 0) {
          (*iterate)[i] = -1.2;
        }
        else {
          (*iterate)[i] = 1.0;
        }
      }
    }

};

class WoodFunction {

  public:
    double Evaluate(const core::table::DensePoint &x) {
      return 100 * core::math::Sqr(x[0] * x[0] - x[1]) +
             core::math::Sqr(1 - x[0]) +
             90 * core::math::Sqr(x[2] * x[2] - x[3]) +
             core::math::Sqr(1 - x[2]) +
             10.1 * (core::math::Sqr(1 - x[1]) + core::math::Sqr(1 - x[3])) +
             19.8 * (1 - x[1]) * (1 - x[3]);
    }

    void Gradient(const core::table::DensePoint &x,
                  core::table::DensePoint *gradient) {
      (*gradient)[0] = 400 * x[0] * (x[0] * x[0] - x[1]) + 2 * (x[0] - 1);
      (*gradient)[1] = 200 * (x[1] - x[0] * x[0]) + 20.2 * (x[1] - 1) +
                       19.8 * (x[3] - 1);
      (*gradient)[2] = 360 * x[2] * (x[2] * x[2] - x[3]) + 2 * (x[2] - 1);
      (*gradient)[3] = 180 * (x[3] - x[2] * x[2]) + 20.2 * (x[3] - 1) +
                       19.8 * (x[1] - 1);
    }

    int num_dimensions() const {
      return 4;
    }

    void InitStartingIterate(core::table::DensePoint *iterate) {

      iterate->Init(num_dimensions());
      (*iterate)[0] = (*iterate)[2] = -3;
      (*iterate)[1] = (*iterate)[3] = -1;
    }

};

class LbfgsTest {
  public:

    void TestExtendedRosenbrockFunction() {

      printf("Testing extended Rosenbrock function: optimal value: 0");
      for(int i = 0; i < 10; i++) {
        core::optimization::lbfgs_test::ExtendedRosenbrockFunction
        extended_rosenbrock_function;
        core::optimization::Lbfgs <
        core::optimization::lbfgs_test::ExtendedRosenbrockFunction >
        extended_rosenbrock_function_lbfgs;
        core::table::DensePoint extended_rosenbrock_function_optimized;
        extended_rosenbrock_function.InitStartingIterate(
          &extended_rosenbrock_function_optimized);
        extended_rosenbrock_function_lbfgs.Init(
          extended_rosenbrock_function,
          std::min(extended_rosenbrock_function.num_dimensions() / 2, 20));
        extended_rosenbrock_function_lbfgs.Optimize(
          -1, &extended_rosenbrock_function_optimized);

        // Test whether the evaluation is close to the zero.
        double function_value = extended_rosenbrock_function.Evaluate(
                                  extended_rosenbrock_function_optimized);
        printf("%d dimensional estended Rosenbrock function optimized to the "
               "function value of %g\n",
               extended_rosenbrock_function.num_dimensions(), function_value);
        if(function_value > 0.5 || function_value < -0.5) {
          throw std::runtime_error("Aborted in extended Rosenbrock test");
        }

        // It should converge to something close to all 1's.
        for(int i = 0; i < extended_rosenbrock_function_optimized.length();
            i++) {
          if(extended_rosenbrock_function_optimized[i] > 1.5 ||
              extended_rosenbrock_function_optimized[i] < 0.5) {
            throw std::runtime_error("Invalid optimal point");
          }
        }
      }
    }

    void TestWoodFunction() {
      printf("Testing wood function: optimal value: 0.\n");
      core::optimization::lbfgs_test::WoodFunction wood_function;
      core::table::DensePoint wood_function_optimized;
      core::optimization::Lbfgs <
      core::optimization::lbfgs_test::WoodFunction > wood_function_lbfgs;
      wood_function.InitStartingIterate(&wood_function_optimized);
      wood_function_lbfgs.Init(wood_function, 2);
      wood_function_lbfgs.Optimize(-1, &wood_function_optimized);

      // It should converge to something close to (1, 1, 1, 1)^T
      for(int i = 0; i < wood_function_optimized.length(); i++) {
        if(wood_function_optimized[i] < 0.5 ||
            wood_function_optimized[i] > 1.5) {
          throw std::runtime_error("Failed in wood function");
        }
      }
    }
};
};
};
};

int main(int argc, char *argv[]) {
  core::table::DensePoint v;
  printf("Starting L-BFGS tests.\n");
  core::optimization::lbfgs_test::LbfgsTest test;
  test.TestExtendedRosenbrockFunction();
  test.TestWoodFunction();
  printf("All tests passed!");
}
