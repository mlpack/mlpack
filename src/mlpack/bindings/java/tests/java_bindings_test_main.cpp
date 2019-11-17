#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/util/cli.hpp>


using namespace std;
using namespace mlpack;

PROGRAM_INFO("Java CLI binding test", "Java CLI binding test", "Some description");

PARAM_STRING_IN("string_in", "Input string, must be 'hello'.", "s", "hello");
PARAM_INT_IN("int_in", "Input int, must be 12.", "i", 3);
PARAM_DOUBLE_IN("double_in", "Input double, must be 4.0.", "d", 0.0);
PARAM_MATRIX_IN("matrix_in", "Input matrix.", "m");

PARAM_STRING_OUT("string_out", "Input string, must be 'hello'.", "S");
PARAM_INT_OUT("int_out", "Input int, must be 12.");
PARAM_DOUBLE_OUT("double_out", "Input double, must be 4.0.");
PARAM_MATRIX_OUT("matrix_out", "Input matrix.", "M");

PARAM_FLAG("flag_in", "Input flag, must be specified.", "f");
PARAM_FLAG("flag_out", "Input flag, must not be specified.", "F");

static void mlpackMain() 
{
    if (CLI::HasParam("string_in"))
    {
        CLI::GetParam<string>("string_out") = CLI::GetParam<string>("string_in") + "_fixed";
    }

    if (CLI::HasParam("int_in"))
    {
        CLI::GetParam<int>("int_out") = CLI::GetParam<int>("int_in") + 4;
    }

    if (CLI::HasParam("matrix_in"))
    {
        CLI::GetParam<arma::mat>("matrix_out") = move(CLI::GetParam<arma::mat>("matrix_in") + 4);
    }

    if (CLI::HasParam("flag_in"))
    {
        CLI::GetParam<bool>("flag_out") = !CLI::GetParam<bool>("flag_in");
    }

    if (CLI::HasParam("double_in"))
    {
        CLI::GetParam<double>("double_out") = CLI::GetParam<double>("double_in") + 4.5;
    }
}
