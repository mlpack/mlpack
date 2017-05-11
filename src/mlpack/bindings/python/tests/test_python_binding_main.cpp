/**
 * @file python_binding_test.cpp
 * @author Ryan Curtin
 *
 * A binding test for Python.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

PROGRAM_INFO("Python binding test",
    "A simple program to test Python binding functionality.");

PARAM_STRING_IN_REQ("string_in", "Input string, must be 'hello'.", "s");
PARAM_INT_IN_REQ("int_in", "Input int, must be 12.", "i");
PARAM_DOUBLE_IN_REQ("double_in", "Input double, must be 4.0.", "d");
PARAM_FLAG("flag1", "Input flag, must be specified.", "f");
PARAM_FLAG("flag2", "Input flag, must not be specified.", "F");
PARAM_MATRIX_IN("matrix_in", "Input matrix.", "m");

PARAM_STRING_OUT("string_out", "Output string, will be 'hello2'.", "S");
PARAM_INT_OUT("int_out", "Output int, will be 13.");
PARAM_DOUBLE_OUT("double_out", "Output double, will be 5.0.");
PARAM_MATRIX_OUT("matrix_out", "Output matrix.", "M");

using namespace std;
using namespace mlpack;

void mlpackMain()
{
  const string s = CLI::GetParam<string>("string_in");
  const int i = CLI::GetParam<int>("int_in");
  const double d = CLI::GetParam<double>("double_in");

  CLI::GetParam<string>("string_out") = "wrong";
  CLI::GetParam<int>("int_out") = 11;
  CLI::GetParam<double>("double_out") = 3.0;

  // Check that everything is right on the input, and then set output
  // accordingly.
  if (!CLI::HasParam("flag2") && CLI::HasParam("flag1"))
  {
    if (s == "hello")
      CLI::GetParam<string>("string_out") = "hello2";

    if (i == 12)
      CLI::GetParam<int>("int_out") = 13;

    if (d == 4.0)
      CLI::GetParam<double>("double_out") = 5.0;
  }

  if (CLI::HasParam("matrix_in"))
  {
    arma::mat out = std::move(CLI::GetParam<arma::mat>("matrix_in"));
    out.shed_row(4);
    out.row(2) *= 2.0;

    CLI::GetParam<arma::mat>("matrix_out") = std::move(out);
  }
}
