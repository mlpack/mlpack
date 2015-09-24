#include "export_filters.hpp"

namespace{

arma::mat const bilinearInterpolation(arma::mat const &src,
                                      size_t height, size_t width)
{
    arma::mat dst(height, width);
    double const x_ratio = static_cast<double>((src.n_cols - 1)) / width;
    double const y_ratio = static_cast<double>((src.n_rows - 1)) / height;
    for(size_t row = 0; row != dst.n_rows; ++row)
    {
        size_t y = static_cast<size_t>(row * y_ratio);
        double const y_diff = (row * y_ratio) - y; //distance of the nearest pixel(y axis)
        double const y_diff_2 = 1 - y_diff;
        for(size_t col = 0; col != dst.n_cols; ++col)
        {
            size_t x = static_cast<size_t>(col * x_ratio);
            double const x_diff = (col * x_ratio) - x; //distance of the nearet pixel(x axis)
            double const x_diff_2 = 1 - x_diff;
            double const y2_cross_x2 = y_diff_2 * x_diff_2;
            double const y2_cross_x = y_diff_2 * x_diff;
            double const y_cross_x2 = y_diff * x_diff_2;
            double const y_cross_x = y_diff * x_diff;
            dst(row, col) = y2_cross_x2 * src(y, x) +
                    y2_cross_x * src(y, x + 1) +
                    y_cross_x2 * src(y + 1, x) +
                    y_cross_x * src(y + 1, x + 1);
        }
    }

    return dst;
}

void copyToExportFilter(arma::mat const &input, arma::mat &output)
{
    int index = 0;
    for(int col = 0; col != output.n_cols; ++col){
        for(int row = 0; row != output.n_rows; ++row){
            output(row, col) = input(index++, 0);
        }
    }
}

}

arma::mat exportFiltersToPGMGrid(std::string const &fileName, arma::mat const &input,
                                 size_t height, size_t width)
{
    arma::mat inputTemp(input);
    double const mean = arma::mean(arma::mean(inputTemp));
    inputTemp -= mean;

    int rows = 0, cols = (int)std::ceil(std::sqrt(inputTemp.n_cols));
    if(std::pow(std::floor(std::sqrt(inputTemp.n_cols)), 2) != inputTemp.n_cols){
        while(inputTemp.n_cols % cols != 0 && cols < 1.2*std::sqrt(inputTemp.n_cols)){
            ++cols;
        }
        rows = (int)std::ceil(inputTemp.n_cols/cols);
    }else{
        cols = (int)std::sqrt(inputTemp.n_cols);
        rows = cols;
    }

    int const SquareRows = (int)std::sqrt(inputTemp.n_rows);
    int const Buf = 1;

    int const Offset = SquareRows+Buf;
    arma::mat array;
    array.ones(Buf+rows*(Offset),
               Buf+cols*(Offset));

    int k = 0;
    for(int i = 0; i != rows; ++i){
        for(int j = 0; j != cols; ++j){
            if(k >= inputTemp.n_cols){
                continue;
            }
            arma::mat reshapeMat(SquareRows, SquareRows);
            copyToExportFilter(inputTemp.col(k), reshapeMat);
            double const max = arma::abs(inputTemp.col(k)).max();
            if(max != 0.0){
                reshapeMat /= max;
            }
            array.submat(i*(Offset), j*(Offset),
                         i*(Offset) + SquareRows - 1,
                         j*(Offset) + SquareRows - 1) =
                    reshapeMat;
            ++k;
        }
    }

    double const max = array.max();
    double const min = array.min();
    if((max - min) != 0){
        array = (array - min) / (max - min) * 255;
    }

    arma::mat result =
            bilinearInterpolation(array, rows * height, cols * width);
    result.save(fileName, arma::pgm_binary);

    return result;
}
