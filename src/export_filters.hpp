#include <armadillo>

arma::mat exportFiltersToPGMGrid(std::string const &fileName,
                                 arma::mat const &input,
                                 size_t height, size_t width);
