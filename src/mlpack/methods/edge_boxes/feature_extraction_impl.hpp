/**
 * @file feature_extraction_impl.hpp
 * @author Nilay Jain
 *
 * Implementation of feature extraction methods.
 */
#ifndef MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_IMPL_HPP
#define MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_IMPL_HPP


#include "feature_extraction.hpp"
#include <map>

namespace mlpack {
namespace structured_tree {

template<typename MatType, typename CubeType>
StructuredForests<MatType, CubeType>::
StructuredForests() 
{
  // to do.
}

template<typename MatType, typename CubeType>
MatType StructuredForests<MatType, CubeType>::
LoadData(MatType const &images, MatType const &boundaries,\
     MatType const &segmentations)
{
  const size_t num_images = this->options["num_images"];
  const size_t row_size = this->options["row_size"];
  const size_t col_size = this->options["col_size"];
  MatType input_data(num_images * row_size * 5, col_size);
  // we store the input data as follows: 
  // images (3), boundaries (1), segmentations (1).
  size_t loop_iter = num_images * 5;
  size_t row_idx = 0;
  size_t col_i = 0, col_s = 0, col_b = 0;
  for(size_t i = 0; i < loop_iter; ++i)
  {
    if (i % 5 == 4)
    {
      input_data.submat(row_idx, 0, row_idx + row_size - 1,\
        col_size - 1) = MatType(segmentations.colptr(col_s),\
                                  col_size, row_size).t();
      ++col_s;
    }
    else if (i % 5 == 3)
    {
      input_data.submat(row_idx, 0, row_idx + row_size - 1,\
        col_size - 1) = MatType(boundaries.colptr(col_b),\
                                  col_size, row_size).t();
      ++col_b;
    }
    else
    {
      input_data.submat(row_idx, 0, row_idx + row_size - 1,\
        col_size - 1) = MatType(images.colptr(col_i),
                                  col_size, row_size).t();
      ++col_i;  
    }
    row_idx += row_size;
  }
  return input_data;
} 

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetFeatureDimension(arma::vec &FtrDim)
{
  /*
  shrink: amount to shrink channels
  p_size: size of image patches
  n_cell: number of self similarity cells
  n_orient: number of orientations per gradient scale
  */
  FtrDim = arma::vec(2);

  const size_t shrink = this->options["shrink"];
  const size_t p_size = this->options["p_size"];
  const size_t n_cell = this->options["n_cell"];
  const size_t rgbd = this->options["rgbd"];
  const size_t n_orient = this->options["n_orient"];
  /*
  n_color_ch: number of color channels
  n_grad_ch: number of gradient channels
  n_ch: total number of channels
  */
  size_t n_color_ch;
  if (this->options["rgbd"] == 0)
    n_color_ch = 3;
  else
    n_color_ch = 4;

  const size_t n_grad_ch = 2 * (1 + n_orient);

  const size_t n_ch = n_color_ch + n_grad_ch;
  FtrDim[0] = std::pow((p_size / shrink) , 2) * n_ch;
  FtrDim[1] = std::pow(n_cell , 2) * (std::pow (n_cell, 2) - 1) / 2 * n_ch;
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
DistanceTransform1D(const arma::vec& f, const size_t n, const double inf,
                    arma::vec& d)
{
  arma::vec v(n), z(n + 1);
  d = arma::vec(n);
  size_t k = 0;
  v[0] = 0.0;
  z[0] = -inf;
  z[1] = inf;
  for (size_t q = 1; q <= n - 1; ++q)
  {
    float s  = ( (f[q] + q * q)-( f[v[k]] + v[k] * v[k]) ) / (2 * q - 2 * v[k]);
    while (s <= z[k])
    {
      --k;
      s  = ( (f[q] + q * q) - (f[v[k]] + v[k] * v[k]) ) / (2 * q - 2 * v[k]);
    }

    ++k;
    v[k] = static_cast<double>(q);
    z[k] = s;
    z[k+1] = inf;
  }

  k = 0;
  for (size_t q = 0; q <= n-1; q++)
  {
    while (z[k+1] < q)
      ++k;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
  return d;
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
DistanceTransform2D(MatType &im, const double inf)
{
  arma::vec f(std::max(im.n_rows, im.n_cols));
  // transform along columns
  for (size_t x = 0; x < im.n_cols; ++x)
  {
    f.subvec(0, im.n_rows - 1) = im.col(x);
    arma::vec d;
    this->DistanceTransform1D(f, im.n_rows, inf, d);
    im.col(x) = d;
  }

  // transform along rows
  for (size_t y = 0; y < im.n_rows; y++)
  {
    f.subvec(0, im.n_cols - 1) = im.row(y).t();
    arma::vec d;
    this->DistanceTransform1D(f, im.n_cols, inf, d);
    im.row(y) = d.t();
  }
}

/* euclidean distance transform of binary image using squared distance */
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
DistanceTransformImage(const MatType& im, double on, MatType& out)
{
  //need a large value but not infinity.
  double inf = 999999.99;
  MatType out = MatType(im.n_rows, im.n_cols, arma::fill::zeros);
  out.elem( find(im != on) ).fill(inf);
  this->DistanceTransform2D(out, inf);
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
CopyMakeBorder(const CubeType& InImage, size_t top, 
               size_t left, size_t bottom, size_t right
               CubeType& OutImage)
{
  OutImage = MatType(InImage.n_rows + top + bottom, InImage.n_cols + left + right, InImage.n_slices);

  for(size_t i = 0; i < InImage.n_slices; ++i)
  {
    OutImage.slice(i).submat(top, left, InImage.n_rows + top - 1, InImage.n_cols + left - 1)
     = InImage.slice(i);
    
    for(size_t j = 0; j < right; ++j)
    {
      OutImage.slice(i).col(InImage.n_cols + left + j).subvec(top, InImage.n_rows + top - 1)
      = InImage.slice(i).col(InImage.n_cols - j - 1);  
    }

    for(size_t j = 0; j < left; ++j)
    {
      OutImage.slice(i).col(j).subvec(top, InImage.n_rows + top - 1)
      = InImage.slice(i).col(left - 1 - j);
    }

    for(size_t j = 0; j < top; j++)
    {

      OutImage.slice(i).row(j)
      = OutImage.slice(i).row(2 * top - 1 - j);
    }
     
    for(size_t j = 0; j < bottom; j++)
    {
      OutImage.slice(i).row(InImage.n_rows + top + j)
      = OutImage.slice(i).row(InImage.n_rows + top - j - 1);
    }

  }
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
RGB2LUV(const CubeType& InImage, CubeType OutImage)
{
  //assert type is double or float.
  double a, y0, maxi;
  a = std::pow(29.0, 3) / 27.0;
  y0 = 8.0 / a;
  maxi = 1.0 / 270.0;

  arma::vec table(1064);
  for (size_t i = 0; i <= 1024; ++i)
  {
    table(i) = i / 1024.0;
    
    if (table(i) > y0)
      table(i) = 116 * pow(table(i), 1.0/3.0) - 16.0;
    else
      table(i) = table(i) * a;

    table(i) = table(i) * maxi;
  }
  for(size_t i = 1025; i < table.n_elem; ++i)
  {
    table(i) = table(i - 1);
  }

  MatType rgb2xyz;
  rgb2xyz << 0.430574 << 0.222015 << 0.020183 << arma::endr
          << 0.341550 << 0.706655 << 0.129553 << arma::endr
          << 0.178325 << 0.071330 << 0.939180;

  //see how to calculate this efficiently. numpy.dot does this.
  CubeType xyz(InImage.n_rows, InImage.n_cols, rgb2xyz.n_cols);

  for (size_t i = 0; i < InImage.slice(0).n_elem; ++i)
  {
    double r = InImage.slice(0)(i);
    double g = InImage.slice(1)(i);
    double b = InImage.slice(2)(i);
    
    xyz.slice(0)(i) = 0.430574 * r + 0.341550 * g + 0.178325 * b;
    xyz.slice(1)(i) = 0.222015 * r + 0.706655 * g + 0.071330 * b;
    xyz.slice(2)(i) = 0.020183 * r + 0.129553 * g + 0.939180 * b;
  
  /*
    xyz.slice(0)(i) = 0.430574 * r + 0.341550 * g + 0.178325 * b;
    xyz.slice(1)(i) = 0.222015 * r + 0.706655 * g + 0.129553 * b;
    xyz.slice(2)(i) = 0.020183 * r + 0.071330 * g + 0.939180 * b;
  */
  }

  MatType nz(InImage.n_rows, InImage.n_cols);

  nz = 1.0 / ( xyz.slice(0) + (15 * xyz.slice(1) ) + 
       (3 * xyz.slice(2) + 1e-35));
  OutImage = MatType(InImage.n_rows, InImage.n_cols, InImage.n_slices);

  for(size_t j = 0; j < xyz.n_cols; ++j)
  {
    for(size_t i = 0; i < xyz.n_rows; ++i)
    {
      OutImage(i, j, 0) = table( static_cast<size_t>( (1024 * xyz(i, j, 1) ) ) );
    }
  }

  OutImage.slice(1) = OutImage.slice(0) % (13 * 4 * (xyz.slice(0) % nz) \
              - 13 * 0.197833) + 88 * maxi;
  OutImage.slice(2) = OutImage.slice(0) % (13 * 9 * (xyz.slice(1) % nz) \
              - 13 * 0.468331) + 134 * maxi;
}

/*implement this function in a column major order.*/
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
BilinearInterpolation(MatType const &src,
                      size_t height, size_t width,
                      MatType dst)
{
  dst = MatType(height, width);
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
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
SepFilter2D(CubeType &InOutImage, arma::vec &kernel, size_t radius)
{
  CubeType OutImage;
  this->CopyMakeBorder(InOutImage, radius, radius, radius, radius, OutImage);

  arma::vec row_res, col_res;
  // reverse InOutImage and OutImage to avoid making an extra matrix.
  // InImage is renamed to InOutImage in this function for this reason only.  
  MatType k_mat = kernel * kernel.t();
  for(size_t k = 0; k < OutImage.n_slices; ++k)
  {
    for(size_t j = radius; j < OutImage.n_cols - radius; ++j)
    {
      for(size_t i = radius; i < OutImage.n_rows - radius; ++i)
      {
        InOutImage(i - radius, j - radius, k) = 
            arma::accu(OutImage.slice(k)\
            .submat(i - radius, j - radius,\
              i + radius, j + radius) % k_mat);
      }
    }
  }

}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
ConvTriangle(CubeType &InImage, size_t radius)
{
  if (radius == 0)
  {
    //nothing to do
  }
  else if (radius <= 1)
  {
    const double p = 12.0 / radius / (radius + 2) - 2;
    arma::vec kernel = {1 , p, 1};
    kernel /= (p + 2);
    
    this->sepFilter2D(InImage, kernel, radius);
  }
  else
  {
    const size_t len = 2 * radius + 1;
    arma::vec kernel(len);
    for( size_t i = 0; i < radius; ++i)
      kernel(i) = i + 1;
    
    kernel(radius) = radius + 1;
    
    size_t r = radius;
    for( size_t i = radius + 1; i < len; ++i)
      kernel(i) = r--;

    kernel /= std::pow(radius + 1, 2);
    this->sepFilter2D(InImage, kernel, radius);
  }
}

//just a helper function, can't use it for anything else
//finds max numbers on cube axis and returns max values,
// also stores the locations of max values in Location
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
MaxAndLoc(CubeType &mag, arma::umat &Location, CubeType& MaxVal) const
{
  /*Vectorize this function after prototype works*/
  MaxVal = MatType(Location.n_rows, Location.n_cols);
  for(size_t i = 0; i < mag.n_rows; ++i)
  {
    for(size_t j = 0; j < mag.n_cols; ++j)
    {
      /*can use -infinity here*/
      double max =  std::numeric_limits<double>::min();
      for(size_t k = 0; k < mag.n_slices; ++k)
      {
        if(mag(i, j, k) > max)
        {
          max = mag(i, j, k);
          MaxVal(i, j) = max;
          Location(i, j) = k;
        }
      }
    }
  }
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
Gradient(const CubeType &InImage, 
         MatType &Magnitude,
         MatType &Orientation)
{
  const size_t grd_norm_rad = this->options["grd_norm_rad"];
  CubeType dx(InImage.n_rows, InImage.n_cols, InImage.n_slices), 
             dy(InImage.n_rows, InImage.n_cols, InImage.n_slices);

  dx.zeros();
  dy.zeros();

  /*
  From MATLAB documentation:
  [FX,FY] = gradient(F), where F is a matrix, returns the 
  x and y components of the two-dimensional numerical gradient. 
  FX corresponds to ∂F/∂x, the differences in x (horizontal) direction. 
  FY corresponds to ∂F/∂y, the differences in the y (vertical) direction.
  */ 


  /*
  gradient calculates the central difference for interior data points.
  For example, consider a matrix with unit-spaced data, A, that has 
  horizontal gradient G = gradient(A). The interior gradient values, G(:,j), are:

  G(:,j) = 0.5*(A(:,j+1) - A(:,j-1));
  where j varies between 2 and N-1, where N is size(A,2).

  The gradient values along the edges of the matrix are calculated with single-sided differences, so that

  G(:,1) = A(:,2) - A(:,1);
  G(:,N) = A(:,N) - A(:,N-1);
  
  The spacing between points in each direction is assumed to be one.
  */
  for (size_t i = 0; i < InImage.n_slices; ++i)
  {
    dx.slice(i).col(0) = InImage.slice(i).col(1) - InImage.slice(i).col(0);
    dx.slice(i).col(InImage.n_cols - 1) = InImage.slice(i).col(InImage.n_cols - 1)
                                        - InImage.slice(i).col(InImage.n_cols - 2);

    for (size_t j = 1; j < InImage.n_cols-1; j++)
      dx.slice(i).col(j) = 0.5 * ( InImage.slice(i).col(j+1) - InImage.slice(i).col(j) ); 

    // do same for dy.
    dy.slice(i).row(0) = InImage.slice(i).row(1) - InImage.slice(i).row(0);
    dy.slice(i).row(InImage.n_rows - 1) = InImage.slice(i).row(InImage.n_rows - 1)
                                        - InImage.slice(i).row(InImage.n_rows - 2);

    for (size_t j = 1; j < InImage.n_rows-1; j++)
      dy.slice(i).row(j) = 0.5 * ( InImage.slice(i).row(j+1) - InImage.slice(i).row(j) );
  }  

  CubeType mag(InImage.n_rows, InImage.n_cols, InImage.n_slices);
  for (size_t i = 0; i < InImage.n_slices; ++i)
  {
    mag.slice(i) = arma::sqrt( arma::square \
                  ( dx.slice(i) + arma::square( dy.slice(i) ) ) );
  }

  arma::umat Location(InImage.n_rows, InImage.n_cols);
  this->MaxAndLoc(mag, Location, Magnitude);
  if(grd_norm_rad != 0)
  {
    //we have to do this ugly thing, or override ConvTriangle
    // and sepFilter2D methods.
    CubeType mag2(InImage.n_rows, InImage.n_cols, 1);
    mag2.slice(0) = Magnitude;
    this->ConvTriangle(mag2, grd_norm_rad);
    Magnitude = Magnitude / (mag2.slice(0) + 0.01);
  }
  MatType dx_mat(dx.n_rows, dx.n_cols),\
            dy_mat(dy.n_rows, dy.n_cols);

  for(size_t j = 0; j < InImage.n_cols; ++j)
  {
    for(size_t i = 0; i < InImage.n_rows; ++i)
    {
      dx_mat(i, j) = dx(i, j, Location(i, j));
      dy_mat(i, j) = dy(i, j, Location(i, j));
    }
  }
  Orientation = arma::atan(dy_mat / dx_mat);
  Orientation.transform( [](double val) { if(val < 0) return (val + arma::datum::pi);  else return (val);} );
  
  for(size_t j = 0; j < InImage.n_cols; ++j)
  {
    for(size_t i = 0; i < InImage.n_rows; ++i)
    {
      if( abs(dx_mat(i, j)) + abs(dy_mat(i, j)) < 1E-5)
        Orientation(i, j) = 0.5 * arma::datum::pi;
    }
  }
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
Histogram(const MatType& Magnitude,
          const MatType& Orientation, 
          size_t downscale, size_t interp,
          CubeType& HistArr)
{
  //i don't think this function can be vectorized.

  //n_orient: number of orientations per gradient scale
  const size_t n_orient = this->options["n_orient"];
  //size of HistArr: n_rbin * n_cbin * n_orient . . . (create in caller...)
  const size_t n_rbin = (Magnitude.n_rows + downscale - 1) / downscale;
  const size_t n_cbin = (Magnitude.n_cols + downscale - 1) / downscale;
  double o_range, o;
  o_range = arma::datum::pi / n_orient;

  HistArr = CubeType(n_rbin, n_cbin, n_orient);
  HistArr.zeros();

  size_t r, c, o1, o2;
  for(size_t i = 0; i < Magnitude.n_rows; ++i)
  {
    for(size_t j = 0; j < Magnitude.n_cols; ++j)
    {
      r = i / downscale;
      c = j / downscale;

      if( interp != 0)
      {
        o = Orientation(i, j) / o_range;
        o1 = ((size_t) o) % n_orient;
        o2 = (o1 + 1) % n_orient;
        HistArr(r, c, o1) += Magnitude(i, j) * (1 + (int)o - o);
        HistArr(r, c, o2) += Magnitude(i, j) * (o - (int) o);
      }
      else
      {
        o1 = (size_t) (Orientation(i, j) / o_range + 0.5) % n_orient;
        HistArr(r, c, o1) += Magnitude(i, j);
      }
    }
  }

  HistArr = HistArr / downscale;

  for (size_t i = 0; i < HistArr.n_slices; ++i)
    HistArr.slice(i) = arma::square(HistArr.slice(i));
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetShrunkChannels(const CubeType& InImage, CubeType &reg_ch, CubeType &ss_ch)
{
  CubeType luv;
  this->RGB2LUV(InImage, luv);
  
  const size_t shrink = this->options["shrink"];
  const size_t n_orient = this->options["n_orient"];
  const size_t grd_smooth_rad = this->options["grd_smooth_rad"];
  const size_t grd_norm_rad = this->options["grd_norm_rad"];
  const size_t num_channels = 13;
  const size_t rsize = luv.n_rows / shrink;
  const size_t csize = luv.n_cols / shrink;
  CubeType channels(rsize, csize, num_channels);
  
  
  size_t slice_idx = 0;
  
  for( slice_idx = 0; slice_idx < luv.n_slices; ++slice_idx)
    this->BilinearInterpolation(luv.slice(slice_idx), (size_t)rsize, (size_t)csize
                                channels.slice(slice_idx));

  double scale = 0.5;
  
  while(scale <= 1.0)
  {
    CubeType img( (luv.n_rows * scale),
                   (luv.n_cols * scale),
                   luv.n_slices );

    for( slice_idx = 0; slice_idx < luv.n_slices; ++slice_idx)
    {
      this->BilinearInterpolation(luv.slice(slice_idx), 
            (luv.n_rows * scale), (luv.n_cols * scale) 
            img.slice(slice_idx));
    }
    
    CubeType OutImage = this->ConvTriangle(img, grd_smooth_rad); 
    
    MatType Magnitude(InImage.n_rows, InImage.n_cols),
             Orientation(InImage.n_rows, InImage.n_cols);
    
    this->Gradient(OutImage, Magnitude, Orientation);
    
    size_t downscale = std::max(1, (int)(shrink * scale));
    
    CubeType Hist = this->Histogram(Magnitude, Orientation,
                                downscale, 0);

    BilinearInterpolation( Magnitude, rsize, csize, channels.slice(slice_idx));
    slice_idx++;
    for(size_t i = 0; i < InImage.n_slices; ++i)
      BilinearInterpolation( Magnitude, rsize, csize,\
                      channels.slice(i + slice_idx));    
    slice_idx += 3;
    scale += 0.5;
  }
  
  //cout << "size of channels: " << arma::size(channels) << endl;
  double reg_smooth_rad, ss_smooth_rad;
  reg_smooth_rad = this->options["reg_smooth_rad"] / (double) shrink;
  ss_smooth_rad = this->options["ss_smooth_rad"] / (double) shrink;




  if (reg_smooth_rad > 1.0)
    reg_ch = this->ConvTriangle(channels, (size_t) (std::round(reg_smooth_rad)) );
  else
    reg_ch = this->ConvTriangle(channels, reg_smooth_rad);

  if (ss_smooth_rad > 1.0)
    ss_ch = this->ConvTriangle(channels, (size_t) (std::round(ss_smooth_rad)) );
  else
    ss_ch = this->ConvTriangle(channels, ss_smooth_rad);

}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
ViewAsWindows(const CubeType& channels, arma::umat const &loc,
              CubeType& features)
{
  // 500 for pos_loc, and 500 for neg_loc.
  // channels = 160, 240, 13.
  features = CubeType(16, 16, 1000 * 13);
  const size_t patchSize = 16;
  const size_t p = patchSize / 2;
  //increase the channel boundary to protect error against image boundaries.
  CubeType inc_ch;
  this->CopyMakeBorder(channels, p, p, p, p, inc_ch);
  for (size_t i = 0, channel = 0; i < loc.n_rows; ++i)
  {
    size_t x = loc(i, 0);
    size_t y = loc(i, 1);

    /*(x,y) in channels, is ((x+p), (y+p)) in inc_ch*/
    CubeType patch = inc_ch.tube((x + p) - p, (y + p) - p,\
                          (x + p) + p - 1, (y + p) + p - 1);
    // since each patch has 13 channel we have to increase the index by 13
    
    //cout <<"patch size = " << arma::size(patch) << endl;
    
    features.slices(channel, channel + 12) = patch;
    channel += 13;
  }
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
Rearrange(CubeType const &channels, CubeType& ch)
{
  //we do (16,16,13*1000) to 256, 1000, 13, in vectorized code.
  ch = CubeType(256, 1000, 13);
  for(size_t i = 0; i < 1000; ++i)
  {
    //MatType m(256, 13);
    for(size_t j = 0; j < 13; ++j)
    {
      size_t sl = (i * j) / 1000;
      //cout << "(i,j) = " << i << ", " << j << endl;
      ch.slice(sl).col(i) = arma::vectorise(channels.slice(i * j));
    }
  }
}

// returns 256 * 1000 * 13 dimension features.
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetRegFtr(const CubeType& channels,const arma::umat& loc
          CubeType& RegFtr)
{
  int shrink = this->options["shrink"];
  int p_size = this->options["p_size"] / shrink;
  CubeType wind;
  this->ViewAsWindows(channels, loc, wind);
  this->Rearrange(wind, RegFtr);
}

template<typename MatType, typename CubeType>
CubeType StructuredForests<MatType, CubeType>::
PDist(const CubeType& features, const arma::uvec& grid_pos,
      CubeType& output)
{
  // size of DestArr: 
  // InImage.n_rows * (InImage.n_rows - 1)/2 * InImage.n_slices
  //find nC2 differences, for locations in the grid_pos.
  //python: input: (716, 256, 13) --->(716, 25, 13) ; output: (716, 300, 13).
  //input features : 256,1000,13; output: 300, 1000, 13

  output = CubeType(300, 1000, 13);
  for(size_t k = 0; k < features.n_slices; ++k)
  {
    size_t r_idx = 0;
    for(size_t i = 0; i < grid_pos.n_elem; ++i) //loop length : 25
    {
      for(size_t j = i + 1; j < grid_pos.n_elem; ++j) //loop length : 25
      {
        output.slice(k).row(r_idx) = features.slice(k).row(grid_pos(i))
                                    - features.slice(k).row(grid_pos(j));
        ++r_idx;
      }
    }
  }
}

//returns 300,1000,13 dimension features.
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetSSFtr(CubeType const &channels, arma::umat const &loc
          CubeType SSFtr)
{
  const size_t shrink = this->options["shrink"];
  const size_t p_size = this->options["p_size"] / shrink;

  //n_cell: number of self similarity cells
  const size_t n_cell = this->options["n_cell"];
  const size_t half_cell_size = (size_t) round(p_size / (2.0 * n_cell));

  arma::uvec g_pos(n_cell);
  for(size_t i = 0; i < n_cell; ++i)
  {
    g_pos(i) = (size_t)round( (i + 1) * (p_size + 2 * half_cell_size \
                     - 1) / (n_cell + 1.0) - half_cell_size);
  }
  arma::uvec grid_pos(n_cell * n_cell);
  size_t k = 0;
  for(size_t i = 0; i < n_cell; ++i)
  {
    for(size_t j = 0; j < n_cell; ++j)
    {
      grid_pos(k) = g_pos(i) * p_size + g_pos(j);
      ++k;
    }
  }

  CubeType wind;
  this->ViewAsWindows(channels, loc, wind);
  CubeType re_wind;
  this->Rearrange(wind, re_wind);
  this->PDist(re_wind, grid_pos, SSFtr); 
}

template<typename MatType, typename CubeType>
void <CubeType> StructuredForests<MatType, CubeType>::
GetFeatures(const MatType &image, arma::umat &loc, 
            CubeType& RegFtr, CubeType& SSFtr)
{
  const size_t row_size = this->options["row_size"];
  const size_t col_size = this->options["col_size"];
  const size_t bottom = (4 - (image.n_rows / 3) % 4) % 4;
  const size_t right = (4 - image.n_cols % 4) % 4;
  //cout << "Botttom = " << bottom << " right = " << right << endl;

  CubeType InImage(image.n_rows / 3, image.n_cols, 3);

  for(size_t i = 0; i < 3; ++i)
  {
    InImage.slice(i) = image.submat(i * row_size, 0, \
                      (i + 1) * row_size - 1, col_size - 1);
  }
  
  CubeType OutImage;
  this->CopyMakeBorder(InImage, 0, 0, bottom, right, OutImage);

  const size_t num_channels = 13;
  const size_t shrink = this->options["shrink"];
  const size_t rsize = OutImage.n_rows / shrink;
  const size_t csize = OutImage.n_cols / shrink;

  /* this part gives double free or corruption out error
     when executed for a second time */
  CubeType reg_ch = CubeType(rsize, csize, num_channels);
  CubeType ss_ch = CubeType(rsize, csize, num_channels);
  this->GetShrunkChannels(InImage, reg_ch, ss_ch);
  
  loc /= shrink;

  this->GetRegFtr(reg_ch, loc, RegFtr);
  this->GetSSFtr(ss_ch, loc, SSFtr);
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
PrepareData(MatType const &InputData)
{
  const size_t num_images = this->options["num_images"];
  const size_t n_tree = this->options["n_tree"];
  const size_t n_pos = this->options["n_pos"];
  const size_t n_neg = this->options["n_neg"];
  const double fraction = 0.25;
  const size_t p_size = this->options["p_size"];
  const size_t g_size = this->options["g_size"];
  const size_t shrink = this->options["shrink"];
  const size_t row_size = this->options["row_size"];
  const size_t col_size = this->options["col_size"];
  // p_rad = radius of image patches.
  // g_rad = radius of ground truth patches.
  const size_t p_rad = p_size / 2, g_rad = g_size / 2;
  
  arma::vec FtrDim;
  this->GetFeatureDimension(FtrDim);
  const size_t n_ftr_dim = FtrDim(0) + FtrDim(1);
  const size_t n_smp_ftr_dim = (size_t)(n_ftr_dim * fraction);

  for(size_t i = 0; i < n_tree; ++i)
  {
    //implement the logic for if data already exists.
    MatType ftrs = arma::zeros(n_pos + n_neg, n_smp_ftr_dim);

    //effectively a 3d array. . .
    MatType lbls = arma::zeros( g_size * g_size, (n_pos + n_neg ));
    // still to be done: store features and labels calculated 
    // in the loop and store it in these Matrices.
    // Could use some suggestions for this.

    size_t loop_iter = num_images * 5;
    for(size_t j = 0; j < loop_iter; j += 5)
    {
      MatType img, bnds, segs;
      img = InputData.submat(j * row_size, 0, (j + 3) * row_size - 1, col_size - 1);
      bnds = InputData.submat( (j + 3) * row_size, 0, \
                        (j + 4) * row_size - 1, col_size - 1 );
      segs = InputData.submat( (j + 4) * row_size, 0, \
                        (j + 5) * row_size - 1, col_size - 1 );

      MatType mask = arma::zeros(row_size, col_size);
      for(size_t b = 0; b < mask.n_cols; b = b + shrink)
        for(size_t a = 0; a < mask.n_rows; a = a + shrink)
          mask(a, b) = 1;
      mask.col(p_rad - 1).fill(0);
      mask.row( (mask.n_rows - 1) - (p_rad - 1) ).fill(0);
      mask.submat(0, 0, mask.n_rows - 1, p_rad - 1).fill(0);
      mask.submat(0, mask.n_cols - p_rad, mask.n_rows - 1, 
                  mask.n_cols - 1).fill(0);

      // number of positive or negative patches per ground truth.
      //int n_patches_per_gt = (int) (ceil( (float)n_pos / num_images ));
      const size_t n_patches_per_gt = 500;
      //cout << "n_patches_per_gt = " << n_patches_per_gt << endl;
      MatType dis;
      this->DistanceTransformImage(bnds, 1, dis)
      dis = arma::sqrt(dis);
      //dis.transform( [](double val, const int& g_rad) { return (double)(val < g_rad); } );
      //dis2.transform( [](double val, const int& g_rad) { return (double)(val >= g_rad); } );
      //dis.elem( arma::find(dis >= g_rad) ).zeros();
      //dis2.elem( arma::find(dis < g_rad) ).zeros();
      

      arma::uvec pos_loc = arma::find( (dis < g_rad) % mask );
      arma::uvec neg_loc = arma::find( (dis >= g_rad) % mask );

      pos_loc = arma::shuffle(pos_loc);
      neg_loc = arma::shuffle(neg_loc);

      arma::umat loc(n_patches_per_gt * 2, 2);
      //cout << "pos_loc size: " << arma::size(pos_loc) << " neg_loc size: " << arma::size(neg_loc) << endl;
      //cout << "n_patches_per_gt = " << n_patches_per_gt << endl;
      for(size_t i = 0; i < n_patches_per_gt; ++i)
      {
        loc.row(i) = arma::ind2sub(arma::size(dis.n_rows, dis.n_cols), pos_loc(i) ).t();
        //cout << "pos_loc: " << loc(i, 0) << ", " << loc(i, 1) << endl;
      }

      for(size_t i = n_patches_per_gt; i < 2 * n_patches_per_gt; ++i)
      {
        loc.row(i) = arma::ind2sub(arma::size(dis.n_rows, dis.n_cols), neg_loc(i) ).t();
        //cout << "neg_loc: " << loc(i, 0) << ", " << loc(i, 1) << endl;
      }
      
      // cout << "num patches = " << n_patches_per_gt << " num elements + = " << pos_loc.n_elem\
      //  << " num elements - = " << neg_loc.n_elem << " dis.size " << dis.n_elem << endl;

      CubeType SSFtr, RegFtr;
      arma::field<CubeType> F = this->GetFeatures(img, loc);
      //randomly sample 70 values each from reg_ftr and ss_ftr.
      /*
      CubeType ftr(140, 1000, 13);
      arma::uvec r = (0, 255, 256);
      arma::uvec s = (0, 299, 300);
      arma::uvec rs = r.shuffle();
      arma::uvec ss = s.shuffle();
      */
      MatType lbl(g_size * g_size, 1000);
      CubeType s(segs.n_rows, segs.n_cols, 1);
      
      // have to do this or we can overload the CopyMakeBorder to support MatType.
      s.slice(0) = segs;
      CubeType in_segs;
      this->CopyMakeBorder(s, g_rad, 
                           g_rad, g_rad, g_rad, in_segs);

      for(size_t i = 0; i < loc.n_rows; ++i)
      {
        size_t x = loc(i, 0); size_t y = loc(i, 1);
        //cout << "x, y = " << x << " " << y << endl;
        lbl.col(i) = arma::vectorise(in_segs.slice(0)\
                    .submat((x + g_rad) - g_rad, (y + g_rad) - g_rad,\
                     (x + g_rad) + g_rad - 1, (y + g_rad) + g_rad - 1));
      }
    }
  }
}
/*
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
Discretize(MatType const &labels, size_t n_class, size_t n_sample)
{
  // Map labels to discrete class labels.
  // lbls : 256 * 20000.
  // n_sample: number of samples for clustering structured labels 256

  // see the return type.
  arma::uvec lis1(n_sample);
  
  MatType zs(n_sample, lbls.n_cols);
  for (size_t i = 0; i < lis1.n_elem; ++i)
    lis1(i) = i;
  MatType DiscreteLabels = arma::zeros(n_sample, n);
  
  for (size_t i = 0; i < labels.n_cols; ++i)
  {
    arma::uvec z1 = lis1.shuffle();
    arma::uvec z2 = lis2.shuffle();
    for (size_t j = 0; j < zs.n_rows; ++i)
      zs(i, j) = (labels(i, z1(j)) == labels(i, z2(j))) ? 1 : 0;
  }
  zs -= arma::mean(zs, 1); // calculate mean about cols. n_col = 256.
  if ( arma::find(zs).n_elem == 0 )
  {
    labels.fill(ones);
  }
  else
  {
    //find most representative segs
  }
  // discretize zs by discretizing pca dimensions
  size_t d = min(5, n_sample, (size_t)floor(log(n_class, 2)));
  zs = pca();

}*/
} // namespace structured_tree
} // namespace mlpack
#endif

