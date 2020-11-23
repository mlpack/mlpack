#ifndef MLPACK_CORE_DATA_DATA_PRE_PROCESSING_HPP
#define MLPACK_CORE_DATA_DATA_PRE_PROCESSING_HPP

#include <mlpack/core.hpp>
#include <fstream>
#include <vector>

namespace mlpack {
namespace data {

class Data {
  private:
    std::string path;
    std::vector<std::string> headers;
    std::vector<bool> numericAttributes;
    arma::mat numericData;

  public:
    arma::mat data;
    Data(std::string file_path);
    void info();
    void getHeaders();
    bool is_digits(const std::string& str);
    void getNumericAttributes();
    void infoHelper(int lower, int upper);
    void head();
    void tail();
    void describe();
	};
}
}

#endif
