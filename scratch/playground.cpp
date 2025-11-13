#include <mlpack/core.hpp>
#include <armadillo>
#include <iostream>

int main()
{
    mlpack::data::ImageInfo info;
    arma::Mat<unsigned char> matrix;

    // executable runs in ~/mlpack/build/bin, so go up to scratch/images
    const std::string inPath  = "../scratch/images/sheep.jpg";
    const std::string outPath = "../scratch/images/sheep-mod.jpg";

    if (!mlpack::data::Load(inPath, matrix, info, true))
    {
        std::cerr << "Failed to load " << inPath << "\n";
        return 1;
    }

    std::cout << "width=" << info.Width()
              << " height=" << info.Height()
              << " channels=" << info.Channels() << "\n";

    // pixel (x=3, y=4), first channel
    const size_t idx =
        4 * info.Width() * info.Channels() + 3 * info.Channels();
    std::cout << "pixel(3,4,c0)=" << (int) matrix[idx] << "\n";

    // simple edit
    matrix += 1;
    matrix = arma::clamp(matrix, (unsigned char)0, (unsigned char)255);

    if (!mlpack::data::Save(outPath, matrix, info))
    {
        std::cerr << "Failed to save " << outPath << "\n";
        return 1;
    }

    std::cout << "wrote " << outPath << "\n";
    return 0;
}
