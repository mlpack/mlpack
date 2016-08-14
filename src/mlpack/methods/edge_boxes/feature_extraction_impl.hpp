/**
 * @file feature_extraction_Impl.hpp
 * @author Nilay Jain
 *
 * Implementation of feature extraction methods.
 */
#ifndef MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_IMPL_HPP
#define MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_IMPL_HPP


#include "feature_extraction.hpp"
#include <mlpack/methods/pca/pca.hpp>

namespace mlpack {
namespace structured_tree {


template<typename MatType, typename CubeType>
StructuredForests<MatType, CubeType>::
StructuredForests(mlpack::structured_tree::FeatureParameters F) : params(F) {}


template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetFeatureDimension(arma::vec& FtrDim)
{
  FtrDim = arma::vec(2);

  const size_t shrink = this->params.Shrink();
  const size_t pSize = this->params.PSize();
  const size_t numCell = this->params.NumCell();
  const size_t numOrient = this->params.NumOrient();
  const size_t nColorCh = params.RGBD() == 0 ? 3 : 4;
  const size_t nCh = nColorCh + 2 * (1 + numOrient);
  FtrDim[0] = std::pow((pSize / shrink) , 2) * nCh;
  FtrDim[1] = std::pow(numCell , 2) * (std::pow (numCell, 2) - 1) / 2 * nCh;
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
}


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


template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
DistanceTransformImage(const MatType& Im, double on, MatType& Out)
{
  //need a large value but not infinity.
  double inf = 999999.99;
  Out = MatType(Im.n_rows, Im.n_cols, arma::fill::zeros);
  Out.elem( find(Im != on) ).fill(inf);
  this->DistanceTransform2D(Out, inf);
}


template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
CopyMakeBorder(const CubeType& InImage, size_t top, 
               size_t left, size_t bottom, size_t right,
               CubeType& OutImage)
{
  OutImage = CubeType(InImage.n_rows + top + bottom, InImage.n_cols + left + right, InImage.n_slices);

  for(size_t i = 0; i < InImage.n_slices; ++i)
  {
    OutImage.slice(i).submat(top, left, InImage.n_rows + top - 1, InImage.n_cols + left - 1)
     = InImage.slice(i);
    
    // first copy borders from left and right 
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

    // copy borders from top and bottom
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
RGB2LUV(const CubeType& InImage, CubeType& OutImage,
        const arma::vec& table)
{

  CubeType xyz(InImage.n_rows, InImage.n_cols, 3);
  
  xyz.slice(0) = 0.430574 * InImage.slice(0) + 0.341550 * InImage.slice(1)
                 + 0.178325 * InImage.slice(2);
  xyz.slice(1) = 0.222015 * InImage.slice(0) + 0.706655 * InImage.slice(1)
                 + 0.071330 * InImage.slice(2);
  xyz.slice(2) = 0.020183 * InImage.slice(0) + 0.129553 * InImage.slice(1)
                 + 0.939180 * InImage.slice(2);
  MatType nz(InImage.n_rows, InImage.n_cols);

  nz = 1.0 / ( xyz.slice(0) + (15 * xyz.slice(1) ) + 
       (3 * xyz.slice(2) + 1e-35));
  OutImage = CubeType(InImage.n_rows, InImage.n_cols, InImage.n_slices);
  for(size_t j = 0; j < xyz.n_cols; ++j)
  {
    for(size_t i = 0; i < xyz.n_rows; ++i)
    {
      OutImage(i, j, 0) = table( static_cast<size_t>( (1024 * xyz(i, j, 1) ) ) );
    }
  }
  double maxi = 1.0 / 270.0;
  OutImage.slice(1) = OutImage.slice(0) % (13 * 4 * (xyz.slice(0) % nz) 
              - 13 * 0.197833) + 88 * maxi;
  OutImage.slice(2) = OutImage.slice(0) % (13 * 9 * (xyz.slice(1) % nz) 
              - 13 * 0.468331) + 134 * maxi;
}


/*Implement this function in a column major order.*/
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
BilinearInterpolation(const MatType& src,
                      const size_t height, 
                      const size_t width,
                      MatType& dst)
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
Convolution(CubeType &InOutImage, const MatType& Filter, const size_t radius)
{
  CubeType OutImage;
  this->CopyMakeBorder(InOutImage, radius, radius, radius, radius, OutImage);

  arma::vec row_res, col_res;
  // reverse InOutImage and OutImage to avoid making an extra matrix.
  // InImage is renamed to InOutImage in this function for this reason only.
  for(size_t k = 0; k < OutImage.n_slices; ++k)
  {
    for(size_t j = radius; j < OutImage.n_cols - radius; ++j)
    {
      for(size_t i = radius; i < OutImage.n_rows - radius; ++i)
      {
        InOutImage(i - radius, j - radius, k) = 
            arma::accu(OutImage.slice(k)
            .submat(i - radius, j - radius,
              i + radius, j + radius) % Filter);
      }
    }
  }

}


template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
ConvTriangle(CubeType &InImage, const size_t radius)
{
  if (radius == 0)
  {
    return;
  }
  else if (radius <= 1)
  {
    const double p = 12.0 / radius / (radius + 2) - 2;
    arma::vec kernel = {1, p, 1};
    kernel /= (p + 2);
    MatType Filter = kernel * kernel.t();
    this->Convolution(InImage, Filter, radius);
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
    MatType Filter = kernel * kernel.t();
    this->Convolution(InImage, Filter, radius);
  }
}


template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
MaxAndLoc(CubeType& mag, arma::umat& Location, MatType& MaxVal) const
{
  /*Vectorize this function after prototype works*/
  MaxVal = MatType(Location.n_rows, Location.n_cols);
  for(size_t i = 0; i < mag.n_rows; ++i)
  {
    for(size_t j = 0; j < mag.n_cols; ++j)
    {
      double max = -DBL_MAX;
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
Gradient(const CubeType& InImage, 
         MatType& Magnitude,
         MatType& Orientation)
{
  const size_t grdNormRad = this->params.GrdNormRad();

  // calculate gradients using sobel filter.
  CubeType dx = InImage; 
  CubeType dy = InImage;

  MatType gx, gy;

  // values for sobel filter.
  gx << -1 << 0 << 1 << arma::endr
     << -2 << 0 << 2 << arma::endr
     << -1 << 0 << 1;

  gy << -1 << -2 << -1 << arma::endr
     << 0 << 0 << 0 << arma::endr
     << 1 << 2 << 1;

  Convolution(dx, gx, 2);
  Convolution(dy, gy, 2);

  // calculate the magnitudes of edges.
  CubeType mag(InImage.n_rows, InImage.n_cols, InImage.n_slices);
  for (size_t i = 0; i < InImage.n_slices; ++i)
  {
    mag.slice(i) = arma::sqrt( arma::square 
                  ( dx.slice(i) + arma::square( dy.slice(i) ) ) );
  }

  arma::umat Location(InImage.n_rows, InImage.n_cols);
  this->MaxAndLoc(mag, Location, Magnitude);
  if(grdNormRad != 0)
  {
    //we have to do this or override ConvTriangle and Convolution methods.
    CubeType mag2(InImage.n_rows, InImage.n_cols, 1);
    mag2.slice(0) = Magnitude;
    this->ConvTriangle(mag2, grdNormRad);
    Magnitude = Magnitude / (mag2.slice(0) + 0.01);
  }

  MatType dx_mat(dx.n_rows, dx.n_cols),
            dy_mat(dy.n_rows, dy.n_cols);

  for(size_t j = 0; j < InImage.n_cols; ++j)
  {
    for(size_t i = 0; i < InImage.n_rows; ++i)
    {
      dx_mat(i, j) = dx(i, j, Location(i, j));
      dy_mat(i, j) = dy(i, j, Location(i, j));
    }
  }
  
  // calculate Orientation of edges.
  Orientation = arma::atan(dy_mat / dx_mat);

  Orientation.transform( [](double val)
   { if(val < 0) return (val + arma::datum::pi);
     else return (val);} );
  
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
          const size_t downscale,
          const size_t interp,
          CubeType& HistArr)
{

  //numOrient: number of orientations per gradient scale
  const size_t numOrient = this->params.NumOrient();
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
  HistArr = arma::square(HistArr);
}

/**
 * Shrink the size of Image by shrink size.
 * Change color space of Image.
 * Extract candidate features.
 * @param InImage, Input Image.
 * @param regCh, 
 * @param ssCh, 
 */

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetShrunkChannels(const CubeType& InImage, CubeType& reg_ch,
                  CubeType& ss_ch, const arma::vec& table)
{
  CubeType luv;
  this->RGB2LUV(InImage, luv, table);
  const size_t shrink = this->params.Shrink();
  const size_t grdSmoothRad = this->params.GrdSmoothRad();
  const size_t numChannels = 13;
  const size_t rsize = luv.n_rows / shrink;
  const size_t csize = luv.n_cols / shrink;
 
  CubeType channels(rsize, csize, numChannels);

  size_t slice_idx;
  for( slice_idx = 0; slice_idx < luv.n_slices; ++slice_idx)
    this->BilinearInterpolation(luv.slice(slice_idx), rsize, csize,
                                channels.slice(slice_idx));
  double scale = 1.0;
  
  while(scale >= 0.5)
  {
    CubeType Img( (luv.n_rows * scale),
                   (luv.n_cols * scale),
                   luv.n_slices );

    for( slice_idx = 0; slice_idx < luv.n_slices; ++slice_idx)
    {
      this->BilinearInterpolation(luv.slice(slice_idx), 
            (luv.n_rows * scale), (luv.n_cols * scale), 
            Img.slice(slice_idx));
    }
    this->ConvTriangle(Img, grdSmoothRad); 
    MatType Magnitude(InImage.n_rows, InImage.n_cols),
             Orientation(InImage.n_rows, InImage.n_cols);
    
    this->Gradient(Img, Magnitude, Orientation);
    size_t downscale = std::max(1, (int)(shrink * scale));
    
    CubeType Hist;
    this->Histogram(Magnitude, Orientation,
                    downscale, 0, Hist);
    BilinearInterpolation( Magnitude, rsize, csize, channels.slice(slice_idx));
    slice_idx++;
    for(size_t i = 0; i < InImage.n_slices; ++i)
      BilinearInterpolation( Magnitude, rsize, csize,
                      channels.slice(i + slice_idx));    
    slice_idx += 3;
    scale -= 0.5;
  }
  
  //cout << "size of channels: " << arma::size(channels) << endl;
  double regSmoothRad, ssSmoothRad;
  regSmoothRad = this->params.RegSmoothRad() / (double) shrink;
  ssSmoothRad = this->params.SSSmoothRad() / (double) shrink;


  reg_ch = channels;
  ss_ch = channels;

  if (regSmoothRad > 1.0)
    this->ConvTriangle(channels, (size_t) (std::round(regSmoothRad)) );

  else
    this->ConvTriangle(channels, regSmoothRad);
  

  if (ssSmoothRad > 1.0)
    this->ConvTriangle(channels, (size_t) (std::round(ssSmoothRad)) );
  
  else
    this->ConvTriangle(channels, ssSmoothRad);

}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
ViewAsWindows(const CubeType& channels, const arma::umat& loc,
              CubeType& features)
{
  // 500 for posLoc, and 500 for negLoc.
  // channels = 160, 240, 13.
  features = CubeType(16, 16, loc.n_rows * 13);
  const size_t patchSize = 16;
  const size_t p = patchSize / 2;
  //increase the channel boundary to protect error against Image boundaries.
  CubeType incCh;
  this->CopyMakeBorder(channels, p, p, p, p, incCh);
  for (size_t i = 0, channel = 0; i < loc.n_rows; ++i)
  {
    size_t x = loc(i, 0);
    size_t y = loc(i, 1);

    /*(x,y) in channels, is ((x+p), (y+p)) in incCh*/
    CubeType patch = incCh.tube((x + p) - p, (y + p) - p,
                          (x + p) + p - 1, (y + p) + p - 1);
    // since each patch has 13 channel we have to increase the index by 13    
    features.slices(channel, channel + 12) = patch;
    channel += 13;
  }
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
Rearrange(const CubeType& channels, CubeType& ch)
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
GetRegFtr(const CubeType& channels, const arma::umat& loc,
          CubeType& RegFtr)
{
//  int pSize = this->params.PSize() / shrink;
  CubeType wind;
  this->ViewAsWindows(channels, loc, wind);
  this->Rearrange(wind, RegFtr);
}

template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
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

//returns (300, 1000, 13) dimension features.
template<typename MatType, typename CubeType>
void StructuredForests<MatType, CubeType>::
GetSSFtr(const CubeType& channels, const arma::umat& loc,
          CubeType& SSFtr)
{
  const size_t shrink = this->params.Shrink();
  const size_t pSize = this->params.PSize() / shrink;

  //numCell: number of self sImilarity cells
  const size_t numCell = this->params.NumCell();
  const size_t half_cell_size = (size_t) round(pSize / (2.0 * numCell));

  arma::uvec g_pos(numCell);
  for(size_t i = 0; i < numCell; ++i)
  {
    g_pos(i) = (size_t)round( (i + 1) * (pSize + 2 * half_cell_size 
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
void StructuredForests<MatType, CubeType>::
GetFeatures(const MatType &Image, arma::umat &loc, 
            CubeType& RegFtr, CubeType& SSFtr, const arma::vec& table)
{
  const size_t rowSize = this->params.RowSize();
  const size_t colSize = this->params.ColSize();
  const size_t bottom = (4 - (Image.n_rows / 3) % 4) % 4;
  const size_t right = (4 - Image.n_cols % 4) % 4;

  CubeType InImage(Image.n_rows / 3, Image.n_cols, 3);

  for(size_t i = 0; i < 3; ++i)
  {
    InImage.slice(i) = Image.submat(i * rowSize, 0, 
                      (i + 1) * rowSize - 1, colSize - 1);
  }
  
  CubeType OutImage;
  this->CopyMakeBorder(InImage, 0, 0, bottom, right, OutImage);


  const size_t numChannels = 13;  
  const size_t shrink = this->params.Shrink();
  const size_t rsize = OutImage.n_rows / shrink;
  const size_t csize = OutImage.n_cols / shrink;

  CubeType reg_ch = CubeType(rsize, csize, numChannels);
  CubeType ss_ch = CubeType(rsize, csize, numChannels);

  this->GetShrunkChannels(InImage, reg_ch, ss_ch, table);

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
PrepareData(const MatType& Images, const MatType& Boundaries,
            const MatType& Segmentations)
{
  // create temporary variables.
  const size_t numImages = this->params.NumImages();
  const size_t numTree = this->params.NumTree();
  const size_t numPos = this->params.NumPos();
  const size_t numNeg = this->params.NumNeg();
  const double fraction = this->params.Fraction();
  const size_t pSize = this->params.PSize();
  const size_t gSize = this->params.GSize();
  const size_t rowSize = this->params.RowSize();
  const size_t colSize = this->params.ColSize();
  // pRad = radius of Image patches.
  // gRad = radius of ground truth patches.
  const size_t pRad = pSize / 2, gRad = gSize / 2;
  arma::vec FtrDim;
  // get the dimensions of the features.
  this->GetFeatureDimension(FtrDim);
  const size_t nFtrDim = FtrDim(0) + FtrDim(1);
  // we only keep a fraction of features.
  const size_t nSmpFtrDim = (size_t)(nFtrDim * fraction);

  for(size_t i = 0; i < numTree; ++i)
  {
    // Implement the logic for if data already exists.
    // this is our new feature dimension.
    MatType ftrs = arma::zeros(numPos + numNeg, nSmpFtrDim);

    // effectively a 3d array.
    MatType lbls = arma::zeros((numPos + numNeg ), gSize * gSize);
    // still to be done: store features and labels calculated 
    // in the loop and store it in these Matrices.
    
    size_t loop_iter = numImages;

    // table is a vector which helps in converting Image from RGB2LUV.
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
      table(i) = table(i - 1);

    size_t col_i = 0, col_s = 0, col_b = 0;
    // process data of each image one by one.
    for(size_t j = 0; j < loop_iter; ++j)
    {
      // these varaibles store image, boundaries and segmentation information
      // for each image in our dataset.
      MatType Img, bnds, segs;

      Img = MatType(Images.colptr(col_i), colSize, rowSize * 3).t() / 255;
      col_i += 3;
      
      bnds = MatType(Boundaries.colptr(col_b), colSize, rowSize).t();
      col_b++;

      segs = MatType(Segmentations.colptr(col_s), colSize, rowSize).t();
      col_s++;

      MatType mask(rowSize, colSize, arma::fill::ones);
      mask.col(pRad - 1).fill(0);
      mask.row( (mask.n_rows - 1) - (pRad - 1) ).fill(0);
      mask.submat(0, 0, mask.n_rows - 1, pRad - 1).fill(0);
      mask.submat(0, mask.n_cols - pRad, mask.n_rows - 1, 
                  mask.n_cols - 1).fill(0);

      // number of positive or negative patches per ground truth.
      const size_t nPatchesPerGt = 500;

      MatType dis;

      // calculate distance transform of image boundary.
      this->DistanceTransformImage(bnds, 1, dis);
      // take square root for euclidean distance transform.
      dis = arma::sqrt(dis);      

      // find positive and negative edge locations using mask.
      arma::uvec posLoc = arma::find( (dis < gRad) % mask );
      arma::uvec negLoc = arma::find( (dis >= gRad) % mask );

      // we take a random permutation of posLoc and negLoc.
      posLoc = arma::shuffle(posLoc);
      negLoc = arma::shuffle(negLoc);

      size_t lenLoc = std::min((int) negLoc.n_elem, std::min((int) nPatchesPerGt,
                              (int) posLoc.n_elem));
      arma::umat loc(lenLoc * 2, 2);

      for(size_t i = 0; i < lenLoc; ++i)
        loc.row(i) = arma::ind2sub(arma::size(dis.n_rows, dis.n_cols), posLoc(i) ).t();
      
      for(size_t i = lenLoc; i < 2 * lenLoc; ++i)
        loc.row(i) = arma::ind2sub(arma::size(dis.n_rows, dis.n_cols), negLoc(i - lenLoc) ).t();
      
      CubeType SSFtr, RegFtr;
      Timer::Start("get_features");

      // calculate the regular and self similarity features of
      // the image Img at locations loc.
      this->GetFeatures(Img, loc, RegFtr, SSFtr, table);
      Timer::Stop("get_features");

      //randomly sample 70 values each from reg_ftr and ss_ftr.
      /*
      CubeType ftr(140, 1000, 13);
      arma::uvec r = (0, 255, 256);
      arma::uvec s = (0, 299, 300);
      arma::uvec rs = r.shuffle();
      arma::uvec ss = s.shuffle();
      */
      //MatType lbl(1000, gSize * gSize);
      CubeType s(segs.n_rows, segs.n_cols, 1);
      
      // have to do this or we can overload the CopyMakeBorder to support MatType.
      s.slice(0) = segs;
      CubeType in_segs;
      // add a padding around the segments.
      this->CopyMakeBorder(s, gRad, gRad, gRad,
                            gRad, in_segs);

      for(size_t i = 0; i < loc.n_rows; ++i)
      {
        size_t x = loc(i, 0); size_t y = loc(i, 1);
        // stores the segments window wise into matrix lbls.
        lbls.row(i) = arma::vectorise(in_segs.slice(0)
                    .submat((x + gRad) - gRad, (y + gRad) - gRad,
                     (x + gRad) + gRad - 1, (y + gRad) + gRad - 1)).t();
      }
    }
    // calculates the discrete labels from segments.
    arma::vec DiscreteLabels;
    size_t x = Discretize(lbls, 2, 256, DiscreteLabels);
  }
}

template<typename MatType, typename CubeType>
size_t StructuredForests<MatType, CubeType>::
IndexMin(arma::vec& k)
{
  double s = k(0); size_t ind = 0;
  for (size_t i = 1; i < k.n_elem; ++i)
  {
    if (k(i) < s)
    {
      s = k(i);
      ind = i;
    }
  }
  return ind;
}
// returns the index of the most representative label, and discretizes structured
// label to discreet classes in matrix subLbls. (this is a vector if nClass = 2)
template<typename MatType, typename CubeType>
size_t StructuredForests<MatType, CubeType>::
Discretize(const MatType& labels, const size_t nClass,
           const size_t nSample, arma::vec& DiscreteLabels)
{
  // Map labels to discrete class labels.
  // lbls : 20000 * 256.
  // nSample: number of samples for clustering structured labels 256
  // nClass: number of classes (clusters) for binary splits. 2
  Timer::Start("other_discretize");

  arma::uvec lis1(nSample);  
  for (size_t i = 0; i < lis1.n_elem; ++i)
    lis1(i) = i;
  
  MatType zs(labels.n_rows, nSample);
  // no. of principal components to keep.
  size_t dim = std::min( 5, std::min( (int)nSample,
              (int)std::floor( std::log2( (int)nClass ) ) ) );
  DiscreteLabels = arma::zeros(labels.n_rows, dim);
  arma::uvec z1 = arma::shuffle(lis1);
  arma::uvec z2 = arma::shuffle(lis1);
  for (size_t j = 0; j < zs.n_cols; ++j)
  {
    for (size_t i = 0; i < zs.n_rows; ++i)
      zs(i, j) = (labels(i, z1(j)) == labels(i, z2(j))) ? 1 : 0;
  }
  for (size_t i = 0; i < zs.n_cols; ++i)
    zs.row(i) -= arma::mean(zs, 0); // calculate mean about rows. n_rows = 20000.
  size_t ind = 0;
  arma::uvec k = arma::find(zs > 0);
  if ( k.n_elem == 0)
  {
    DiscreteLabels.ones();
  }
  else
  {
    //find most representative label (closest to mean)
    arma::vec k = arma::sum(arma::abs(zs), 0).t();
    ind = IndexMin(k);
    // so most representative label is: labels.row(ind).

    // apply pca
    Timer::Stop("other_discretize");
    Timer::Start("pca_timer");
    MatType coeff, transformedData;
    arma::vec eigVal;
    mlpack::pca::PCA p;
    p.Apply(zs.t(), transformedData, eigVal, coeff);
    // we take only first row in transformedData (256 * 20000) as dim = 1.
    Timer::Stop("pca_timer");
    Timer::Start("other_discretize");
    DiscreteLabels = arma::conv_to<arma::vec>::from(transformedData.row(0).t() > 0);
    Timer::Stop("other_discretize");
  }
  return ind;

}
} // namespace structured_tree
} // namespace mlpack
#endif


