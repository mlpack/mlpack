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


void resize(const unsigned char* image_data, int s_width, int s_height,
    unsigned char*& frame_buffer_out, int d_width, int d_height, int num_channel)
{
  stbir_resize_uint8(image_data, s_width, s_height, 0,
                     frame_buffer_out, d_width, d_height, 0, num_channel);
}


void resize_and_crop(const unsigned char* image_data, int s_width, int s_height,
    unsigned char*& frame_buffer_out, int d_width, int d_height, int num_channel)
{
  float ratio_w = static_cast<float>(d_width)  / static_cast<float>(s_width);
  float ratio_h = static_cast<float>(d_height) / static_cast<float>(s_height);

  float largest_ratio = ratio_w > ratio_h ? ratio_w : ratio_h;
  //std::cout << "largest_ratio: "<< largest_ratio << std::endl;
  int new_width = static_cast<int>(largest_ratio * s_width);
  int new_height = static_cast<int>(largest_ratio * s_height);

  //std::cout << "new width: "<< new_width << std::endl;
  //std::cout << "new_height: "<< new_height << std::endl;
  unsigned char* buffer_out = 
    (unsigned char*)malloc(new_width * new_height * num_channel * sizeof (unsigned char));
  stbir_resize_uint8(image_data, s_width, s_height, 0,
                     buffer_out, new_width, new_height, 0, num_channel);

  int n_cols_crop = new_width > new_height ? (new_width - new_height) : 0;
  int n_rows_crop = new_height > new_width ? (new_height - new_width) : 0;

  //std::cout << "num cols to crop" << n_cols_crop << std::endl;
  //std::cout << "num rows to crop" << n_rows_crop << std::endl;

  if (n_rows_crop != 0)
  {
    int crop_up_down_equally = (n_rows_crop / 2) * num_channel * new_width;
    arma::Col<unsigned char> matrix(buffer_out, new_height * num_channel * new_width, true, true);
    arma::Col<unsigned char> vec = matrix.subvec(crop_up_down_equally,
        matrix.n_rows - crop_up_down_equally - 1);
    stbir_resize_uint8(vec.memptr(), d_width, d_height, 0,
                       frame_buffer_out, d_width, d_height, 0, num_channel);

    //memcpy(frame_buffer_out, vec.memptr(), d_height * d_width * num_channel);
  }
  else if (n_cols_crop !=0)
  {
    arma::Cube<unsigned char> cube(new_height, new_width, num_channel);
    size_t k = 0;
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
    memcpy(frame_buffer_out, vec.memptr(), d_height * d_width * num_channel);
    //std::cout << "show the memory" << std::endl; 
  }
  else
  {
    //std::cout << "should not assign here" << std::endl;
    memcpy(frame_buffer_out, buffer_out, d_height * d_width * num_channel);
  }
  free(buffer_out);
}



} // namespace data
} // namespace mlpack

#endif

