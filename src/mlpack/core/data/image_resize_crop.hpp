/**
 * @file core/data/image_resize_crop.hpp
 * @author Omar Shrit
 *
 * Image resize and crop functionalities.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_RESIZE_CROP_HPP
#define MLPACK_CORE_DATA_IMAGE_RESIZE_CROP_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {


/**
 * Image resize/crop interfaces.
 */

/**
 * Resize the images matrix.
 *
 * @param images The input matrix that contains the images to be resized.
 * @param info Contains relevant input images information.
 * @param resizedImages The output matrix the contains the resized images.
 * @param newWidth The new requested width for the resized images.
 * @param newHeight The new requested height for the resized images.
 */
template<typename eT>
void ResizeImages(arma::Mat<eT>& images, data::ImageInfo& info,
    arma::Mat resizedImages, const size_t newWidth, const size_t newHeight)
{
  if (images.n_rows != resizedImages and images.n_cols != resizedImages.n_cols)
  {
    std::ostringstream oss;
    oss << "ResizeImage(): the resizedImage matrix need to have identical"
      " dimensions to the images matrix" << std::endl;
  }

  for (size_t i = 0; i < images.n_cols; ++i)
  {
    stbir_resize_uint8(image.colptr(i), info.Width(), info.Height(), 0,
                       resizedImages.colptr(i), newWidth, newHeight, 0,
                       info.Channels());
  }
}

/**
 * This function resizes and crop the images in order to keep the aspect ratio
 * and allows to keep the area of interest of the image. Assuming that the
 * area of interest is usually in the center.
 *
 * Most of images have not equal dimensions, in this function we identify which
 * one has the largest ratio between these dimensions, and crop the difference
 * between these two dimensions and then resize the image to the requested one.
 *
 * This is required to keep high usable resolution, since most 
 * object detection / classification models require equal and low dimensions.
 * If this is not necessary you can use the resize function only.
 * 
 * @param images The input matrix that contains the images to be resized.
 * @param info Contains relevant input images information.
 * @param resizedImages The output matrix the contains the resized images.
 * @param newWidth The new requested width for the resized images.
 * @param newHeight The new requested height for the resized images.
 */


void (const unsigned char* image_data, int s_width, int s_height,
    unsigned char*& frame_buffer_out, int d_width, int d_height, int num_channel)

template<typename eT>
void CropResizeImages(arma::Mat<eT>& images, data::ImageInfo& info,
    arma::Mat resizedImages, const size_t newWidth, const size_t newHeight)
{
  float ratioW = static_cast<float>(newWidth)  /
    static_cast<float>(info.Width());
  float ratioH = static_cast<float>(newHeight) /
    static_cast<float>(info.Height());

  float largestRatio = ratioW > ratioH ? ratioW : ratioH;
  //std::cout << "largest_ratio: "<< largest_ratio << std::endl;
  int tempWidth = static_cast<int>(largestRatio * info.Width());
  int tempHeight = static_cast<int>(largestRatio * info.Height());

  arma::Mat<eT> tempMat(size(images));
  ResizeImage(images, info, tempMat, tempWidth, tempHeight);
 
  int nColsCrop = tempWidth > tempHeight ? (tempWidth - tempHeight) : 0;
  int nRowsCrop = tempHeight > tempWidth ? (tempHeight - tempWidth) : 0;

  //std::cout << "num cols to crop" << n_cols_crop << std::endl;
  //std::cout << "num rows to crop" << n_rows_crop << std::endl;

  if (nRowsCrop != 0)
  {
    int CropUpDownEqually = (nRowsCrop / 2) * info.Channel() * tempWidth; 
    for (size_t i = 0; i < tempMat.n_cols; ++i)
    {
      // Make vec as a vec of images. resize it before the loop
      arma::Col<eT> vec = tempMat.col(i).subvec(CropUpDownEqually,
        temMat.n_rows - CropUpDownEqually - 1);
    }
    ResizeImage(vec, /* info object but with vec information */, resizedImages, newWidth, newHeight);
  }
  else if (nColsCrop !=0)
  {
    arma::Cube<unsigned char> cube(new_height, new_width, num_channel);
    size_t k = 0;
    // It seems that using OpenMP is causing problems in the image, where
    // pixels are not copying in the order that it should be.
    //#pragma omp parallel for collapse(3)
    for (size_t r = 0; r < cube.n_rows; ++r)
    {
      for (size_t c = 0; c < cube.n_cols; ++c)
      {
        for (size_t i = 0; i < cube.n_slices; ++i)
        {
          cube.at(r, c, i) = buffer_out[k];
          k++;
        }
      }
    }
    int rounded = std::round(n_cols_crop / 2);
    //std::cout << "rounded: " << rounded << std::endl;
    cube.shed_cols(0, rounded - 1);
    cube.shed_cols(cube.n_cols - rounded, cube.n_cols - 1);
    k = 0;
    arma::Col<unsigned char> vec(cube.n_cols * cube.n_rows * cube.n_slices);
    //cube.brief_print();
    //#pragma omp parallel for collapse(3)
    for (size_t c = 0; c < cube.n_rows; ++c)
    {
      for (size_t j = 0; j < cube.n_cols; ++j)
      {
        for (size_t i = 0; i < cube.n_slices; ++i)
        {
          vec.at(k) = cube.at(c,j,i);
          k++;
        }
      }
    }
    //cube.brief_print();
    //stbir_resize_uint8(vec.memptr(), cube.n_cols, cube.n_rows, 0,
                       //frame_buffer_out, d_width, d_height, 0, num_channel);
   // memcpy(frame_buffer_out, vec.memptr(), d_height * d_width * num_channel);
    //std::cout << "show the memory" << std::endl; 
  }
  else
  {
    //std::cout << "should not assign here" << std::endl;
    memcpy(frame_buffer_out, buffer_out, d_height * d_width * num_channel);
  }
}

} // namespace data
} // namespace mlpack

#endif

