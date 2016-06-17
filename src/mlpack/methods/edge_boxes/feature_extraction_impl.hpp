/**
 * @file feature_extraction_Impl.hpp
 * @author Nilay Jain
 *
 * Implementation of feature extraction methods.
 */
#ifndef MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_ImPL_HPP
#define MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_ImPL_HPP


#include "feature_extraction.hpp"
#include <map>

namespace mlpack {
namespace structured_tree {

/**
 * Constructor: stores all the parameters in an object
 * of feature_parameters class.
 */
template<typename MatType, typename CubeType>
StructuredForests<MatType, CubeType>::
StructuredForests(FeatureParameters F) 
{
  // to do.
  params = F; 
  //check if this works. 
  std::cout << params.numImages << std::endl;
}

/*
template<typename MatType, typename CubeType>
MatType StructuredForests<MatType, CubeType>::
LoadData(MatType const &Images, MatType const &boundaries,\
     MatType const &segmentations)
{
  const size_t num_Images = this->params.num_Images;
  const size_t rowSize = this->params.rowSize;
  const size_t colSize = this->params.colSize;
  MatType input_data(num_Images * rowSize * 5, colSize);
  // we store the input data as follows: 
  // Images (3), boundaries (1), segmentations (1).
  size_t loop_iter = num_Images * 5;
  size_t row_idx = 0;
  size_t col_i = 0, col_s = 0, col_b = 0;
  for(size_t i = 0; i < loop_iter; ++i)
  {
    if (i % 5 == 4)
    {
      input_data.submat(row_idx, 0, row_idx + rowSize - 1,\
        colSize - 1) = MatType(segmentations.colptr(col_s),\
                                  colSize, rowSize).t();
      ++col_s;
    }
    else if (i % 5 == 3)
    {
      input_data.submat(row_idx, 0, row_idx + rowSize - 1,\
        colSize - 1) = MatType(boundaries.colptr(col_b),\
                                  colSize, rowSize).t();
      ++col_b;
    }
    else
    {
      input_data.submat(row_idx, 0, row_idx + rowSize - 1,\
        colSize - 1) = MatType(Images.colptr(col_i),
                                  colSize, rowSize).t();
      ++col_i;  
    }
    row_idx += rowSize;
  }
  return input_data;
} 
*/

/**
 * Get DImensions of Features
 * @param FtrDIm Output vector that contains the result 
 */
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetFeatureDImension(arma::vec &FtrDIm)
{
  FtrDIm = arma::vec(2);

  const size_t shrink = this->params.shrink;
  const size_t pSize = this->params.pSize;
  const size_t numCell = this->params.numCell;
  const size_t rgbd = this->params.rgbd;
  const size_t numOrient = this->params.numOrient;
  
  size_t nColorCh;
  if (this->params.rgbd == 0)
    nColorCh = 3;
  else
    nColorCh = 4;

  const size_t nCh = nColorCh + 2 * (1 + numOrient);
  FtrDIm[0] = std::pow((pSize / shrink) , 2) * nCh;
  FtrDIm[1] = std::pow(numCell , 2) * (std::pow (numCell, 2) - 1) / 2 * nCh;
}

/**
 * Computes distance transform of 1D vector f.
 * @param f input vector whose distance transform is to be found.
 * @param n size of the Output vector to be made.
 * @param inf a large double value.
 * @param d Output vector which stores distance transform of f.
 */
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

/**
 * Computes distance transform of a 2D array
 * @param Im input array whose distance transform is to be found.
 * @param inf a large double value.
 */

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
DistanceTransform2D(MatType &Im, const double inf)
{
  arma::vec f(std::max(Im.n_rows, Im.n_cols));
  // transform along columns
  for (size_t x = 0; x < Im.n_cols; ++x)
  {
    f.subvec(0, Im.n_rows - 1) = Im.col(x);
    arma::vec d;
    this->DistanceTransform1D(f, Im.n_rows, inf, d);
    Im.col(x) = d;
  }

  // transform along rows
  for (size_t y = 0; y < Im.n_rows; y++)
  {
    f.subvec(0, Im.n_cols - 1) = Im.row(y).t();
    arma::vec d;
    this->DistanceTransform1D(f, Im.n_cols, inf, d);
    Im.row(y) = d.t();
  }
}

/**
 * euclidean distance transform of binary Image using squared distance
 * @param Im Input binary Image whose distance transform is to be found.
 * @param on if on == 1, 1 is taken as boundaries and vice versa.
 * @param Out Output Image.
 */
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
DistanceTransformImage(const MatType& Im, double on, MatType& Out)
{
  //need a large value but not infinity.
  double inf = 999999.99;
  MatType Out = MatType(Im.n_rows, Im.n_cols, arma::fill::zeros);
  Out.elem( find(Im != on) ).fill(inf);
  this->DistanceTransform2D(Out, inf);
}

/**
 * Makes a reflective border around an Image.
 * @param InImage Image which we have to make border around.
 * @param top border length at top.
 * @param left border length at left.
 * @param bottom border length at bottom.
 * @param right border length at right.
 * @param OutImage Output Image. 
 */
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

/**
 * Converts an Image in RGB color space to LUV color space.
 * @param InImage Input Image in RGB color space.
 * @param OutImage Ouptut Image in LUV color space.
 */
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

/**
 * Resizes the Image to the given size using Bilinear Interpolation
 * @param src Input Image
 * @param height Height of Output Image.
 * @param width Width Out Output Image.
 * @param dst Output Image resized to (height, width)
 */
/*Implement this function in a column major order.*/
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

/**
 * Applies a separable linear filter to an Image
 * @param InOutImage Input/Output Contains the input Image, The final filtered Image is
 *          stored in this param.
 * @param kernel Input Kernel vector to be applied on Image.
 * @param radius amount, the Image should be padded before applying filter.
 */
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
SepFilter2D(CubeType &InOutImage, const arma::vec& kernel, const size_t radius)
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

/**
 * Applies a triangle filter on an Image.
 * @param InImage Input/Output Image on which filter is applied.
 * @param radius Decides the size of kernel to be applied on Image.
 */
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
ConvTriangle(CubeType &InImage, const size_t radius)
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
      double max =  std::numeric_lImits<double>::min();
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
  const size_t grdNormRad = this->params.grdNormRad;
  CubeType dx(InImage.n_rows, InImage.n_cols, InImage.n_slices), 
             dy(InImage.n_rows, InImage.n_cols, InImage.n_slices);

  dx.zeros();
  dy.zeros();

  /*
  From MATLAB documentation:
  [FX,FY] = gradient(F), where F is a matrix, returns the 
  x and y components of the two-dImensional numerical gradient. 
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
  if(grdNormRad != 0)
  {
    //we have to do this ugly thing, or override ConvTriangle
    // and sepFilter2D methods.
    CubeType mag2(InImage.n_rows, InImage.n_cols, 1);
    mag2.slice(0) = Magnitude;
    this->ConvTriangle(mag2, grdNormRad);
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

  //numOrient: number of orientations per gradient scale
  const size_t numOrient = this->params.numOrient;
  //size of HistArr: n_rbin * n_cbin * numOrient . . . (create in caller...)
  const size_t n_rbin = (Magnitude.n_rows + downscale - 1) / downscale;
  const size_t n_cbin = (Magnitude.n_cols + downscale - 1) / downscale;
  double o_range, o;
  o_range = arma::datum::pi / numOrient;

  HistArr = CubeType(n_rbin, n_cbin, numOrient);
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
        o1 = ((size_t) o) % numOrient;
        o2 = (o1 + 1) % numOrient;
        HistArr(r, c, o1) += Magnitude(i, j) * (1 + (int)o - o);
        HistArr(r, c, o2) += Magnitude(i, j) * (o - (int) o);
      }
      else
      {
        o1 = (size_t) (Orientation(i, j) / o_range + 0.5) % numOrient;
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
  
  const size_t shrink = this->params.shrink;
  const size_t numOrient = this->params.numOrient;
  const size_t grdSmoothRad = this->params.grdSmoothRad;
  const size_t grdNormRad = this->params.grdNormRad;
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
    CubeType Img( (luv.n_rows * scale),
                   (luv.n_cols * scale),
                   luv.n_slices );

    for( slice_idx = 0; slice_idx < luv.n_slices; ++slice_idx)
    {
      this->BilinearInterpolation(luv.slice(slice_idx), 
            (luv.n_rows * scale), (luv.n_cols * scale) 
            Img.slice(slice_idx));
    }
    
    CubeType OutImage = this->ConvTriangle(Img, grdSmoothRad); 
    
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
  double regSmoothRad, ssSmoothRad;
  regSmoothRad = this->params.regSmoothRad / (double) shrink;
  ssSmoothRad = this->params.ssSmoothRad / (double) shrink;




  if (regSmoothRad > 1.0)
    reg_ch = this->ConvTriangle(channels, (size_t) (std::round(regSmoothRad)) );
  else
    reg_ch = this->ConvTriangle(channels, regSmoothRad);

  if (ssSmoothRad > 1.0)
    ss_ch = this->ConvTriangle(channels, (size_t) (std::round(ssSmoothRad)) );
  else
    ss_ch = this->ConvTriangle(channels, ssSmoothRad);

}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
ViewAsWindows(const CubeType& channels, arma::umat const &loc,
              CubeType& features)
{
  // 500 for posLoc, and 500 for negLoc.
  // channels = 160, 240, 13.
  features = CubeType(16, 16, 1000 * 13);
  const size_t patchSize = 16;
  const size_t p = patchSize / 2;
  //increase the channel boundary to protect error against Image boundaries.
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

// returns 256 * 1000 * 13 dImension features.
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetRegFtr(const CubeType& channels,const arma::umat& loc
          CubeType& RegFtr)
{
  int shrink = this->params.shrink;
  int pSize = this->params.pSize / shrink;
  CubeType wind;
  this->ViewAsWindows(channels, loc, wind);
  this->Rearrange(wind, RegFtr);
}

template<typename MatType, typename CubeType>
CubeType StructuredForests<MatType, CubeType>::
PDist(const CubeType& features, const arma::uvec& grid_pos,
      CubeType& Output)
{
  // size of DestArr: 
  // InImage.n_rows * (InImage.n_rows - 1)/2 * InImage.n_slices
  //find nC2 differences, for locations in the grid_pos.
  //python: input: (716, 256, 13) --->(716, 25, 13) ; Output: (716, 300, 13).
  //input features : 256,1000,13; Output: 300, 1000, 13

  Output = CubeType(300, 1000, 13);
  for(size_t k = 0; k < features.n_slices; ++k)
  {
    size_t r_idx = 0;
    for(size_t i = 0; i < grid_pos.n_elem; ++i) //loop length : 25
    {
      for(size_t j = i + 1; j < grid_pos.n_elem; ++j) //loop length : 25
      {
        Output.slice(k).row(r_idx) = features.slice(k).row(grid_pos(i))
                                    - features.slice(k).row(grid_pos(j));
        ++r_idx;
      }
    }
  }
}

//returns 300,1000,13 dImension features.
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetSSFtr(CubeType const &channels, arma::umat const &loc
          CubeType SSFtr)
{
  const size_t shrink = this->params.shrink;
  const size_t pSize = this->params.pSize / shrink;

  //numCell: number of self sImilarity cells
  const size_t numCell = this->params.numCell;
  const size_t half_cell_size = (size_t) round(pSize / (2.0 * numCell));

  arma::uvec g_pos(numCell);
  for(size_t i = 0; i < numCell; ++i)
  {
    g_pos(i) = (size_t)round( (i + 1) * (pSize + 2 * half_cell_size \
                     - 1) / (numCell + 1.0) - half_cell_size);
  }
  arma::uvec grid_pos(numCell * numCell);
  size_t k = 0;
  for(size_t i = 0; i < numCell; ++i)
  {
    for(size_t j = 0; j < numCell; ++j)
    {
      grid_pos(k) = g_pos(i) * pSize + g_pos(j);
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
GetFeatures(const MatType &Image, arma::umat &loc, 
            CubeType& RegFtr, CubeType& SSFtr)
{
  const size_t rowSize = this->params.rowSize;
  const size_t colSize = this->params.colSize;
  const size_t bottom = (4 - (Image.n_rows / 3) % 4) % 4;
  const size_t right = (4 - Image.n_cols % 4) % 4;
  //cout << "Botttom = " << bottom << " right = " << right << endl;

  CubeType InImage(Image.n_rows / 3, Image.n_cols, 3);

  for(size_t i = 0; i < 3; ++i)
  {
    InImage.slice(i) = Image.submat(i * rowSize, 0, \
                      (i + 1) * rowSize - 1, colSize - 1);
  }
  
  CubeType OutImage;
  this->CopyMakeBorder(InImage, 0, 0, bottom, right, OutImage);

  const size_t num_channels = 13;
  const size_t shrink = this->params.shrink;
  const size_t rsize = OutImage.n_rows / shrink;
  const size_t csize = OutImage.n_cols / shrink;

  /* this part gives double free or corruption Out error
     when executed for a second tIme */
  CubeType reg_ch = CubeType(rsize, csize, num_channels);
  CubeType ss_ch = CubeType(rsize, csize, num_channels);
  this->GetShrunkChannels(InImage, reg_ch, ss_ch);
  
  loc /= shrink;

  this->GetRegFtr(reg_ch, loc, RegFtr);
  this->GetSSFtr(ss_ch, loc, SSFtr);
}

/**
 * This functions prepares the data, 
 * and extracts features, structured labels.
 * @param: 
 */

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
PrepareData(const MatType& Images, const MatType& Boundaries,\
            const MatType& Segmentations)
{
  const size_t numImages = this->params.numImages;
  const size_t numTree = this->params.numTree;
  const size_t numPos = this->params.numPos;
  const size_t numNeg = this->params.numNeg;
  const double fraction = this->params.fraction;
  const size_t pSize = this->params.pSize;
  const size_t gSize = this->params.gSize;
  const size_t shrink = this->params.shrink;
  const size_t rowSize = this->params.rowSize;
  const size_t colSize = this->params.colSize;
  // pRad = radius of Image patches.
  // gRad = radius of ground truth patches.
  const size_t pRad = pSize / 2, gRad = gSize / 2;
  
  arma::vec FtrDIm;
  this->GetFeatureDImension(FtrDIm);
  const size_t nFtrDIm = FtrDIm(0) + FtrDIm(1);
  const size_t nSmpFtrDIm = (size_t)(nFtrDIm * fraction);

  for(size_t i = 0; i < numTree; ++i)
  {
    //Implement the logic for if data already exists.
    MatType ftrs = arma::zeros(numPos + numNeg, nSmpFtrDIm);

    //effectively a 3d array. . .
    MatType lbls = arma::zeros( gSize * gSize, (numPos + numNeg ));
    // still to be done: store features and labels calculated 
    // in the loop and store it in these Matrices.
    // Could use some suggestions for this.

    size_t loop_iter = num_Images;
    for(size_t j = 0; j < loop_iter; ++j)
    {
      MatType Img, bnds, segs;
      Img = Images.submat(j * rowSize, 0, (j + 3) * rowSize - 1, colSize - 1);
      bnds = Boundaries.submat( j * rowSize, 0, \
                        j * rowSize - 1, colSize - 1 );
      segs = Segmentations.submat( j * rowSize, 0, \
                        j * rowSize - 1, colSize - 1 );

      MatType mask(rowSize, colSize, arma::fill::ones);
      mask.col(pRad - 1).fill(0);
      mask.row( (mask.n_rows - 1) - (pRad - 1) ).fill(0);
      mask.submat(0, 0, mask.n_rows - 1, pRad - 1).fill(0);
      mask.submat(0, mask.n_cols - pRad, mask.n_rows - 1, 
                  mask.n_cols - 1).fill(0);

      // number of positive or negative patches per ground truth.

      const size_t nPatchesPerGt = 500;
      MatType dis;
      this->DistanceTransformImage(bnds, 1, dis)
      dis = arma::sqrt(dis);      

      arma::uvec posLoc = arma::find( (dis < gRad) % mask );
      arma::uvec negLoc = arma::find( (dis >= gRad) % mask );

      posLoc = arma::shuffle(posLoc);
      negLoc = arma::shuffle(negLoc);

      arma::umat loc(nPatchesPerGt * 2, 2);

      for(size_t i = 0; i < nPatchesPerGt; ++i)
      {
        loc.row(i) = arma::ind2sub(arma::size(dis.n_rows, dis.n_cols), posLoc(i) ).t();
        //cout << "posLoc: " << loc(i, 0) << ", " << loc(i, 1) << endl;
      }

      for(size_t i = nPatchesPerGt; i < 2 * nPatchesPerGt; ++i)
      {
        loc.row(i) = arma::ind2sub(arma::size(dis.n_rows, dis.n_cols), negLoc(i - nPatchesPerGt) ).t();
      }
      
      CubeType SSFtr, RegFtr;
      this->GetFeatures(Img, loc, RegFtr, SSFtr);
      //randomly sample 70 values each from reg_ftr and ss_ftr.
      /*
      CubeType ftr(140, 1000, 13);
      arma::uvec r = (0, 255, 256);
      arma::uvec s = (0, 299, 300);
      arma::uvec rs = r.shuffle();
      arma::uvec ss = s.shuffle();
      */
      MatType lbl(gSize * gSize, 1000);
      CubeType s(segs.n_rows, segs.n_cols, 1);
      
      // have to do this or we can overload the CopyMakeBorder to support MatType.
      s.slice(0) = segs;
      CubeType in_segs;
      this->CopyMakeBorder(s, gRad, 
                           gRad, gRad, gRad, in_segs);

      for(size_t i = 0; i < loc.n_rows; ++i)
      {
        size_t x = loc(i, 0); size_t y = loc(i, 1);
        //cout << "x, y = " << x << " " << y << endl;
        lbl.col(i) = arma::vectorise(in_segs.slice(0)\
                    .submat((x + gRad) - gRad, (y + gRad) - gRad,\
                     (x + gRad) + gRad - 1, (y + gRad) + gRad - 1));
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
  zs -= arma::mean(zs, 1); // calculate mean abOut cols. n_col = 256.
  if ( arma::find(zs).n_elem == 0 )
  {
    labels.fill(ones);
  }
  else
  {
    //find most representative segs
  }
  // discretize zs by discretizing pca dImensions
  size_t d = min(5, n_sample, (size_t)floor(log(n_class, 2)));
  zs = pca();

}*/
} // namespace structured_tree
} // namespace mlpack
#endif

