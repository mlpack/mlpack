
#ifndef __2D_FACTOR_H
#define __2D_FACTOR_H
#include <fastlib/fastlib.h>

namespace la {

  void row2dPCA(const ArrayList<Matrix>& imageList, 
		Vector& eigenValues, Matrix& rowBasis_out, Matrix& mean_out);

  void col2dPCA(const ArrayList<Matrix>& imageList, 
		Vector& eigenValues, Matrix& colBasis_out, Matrix& mean_out);

  void RowCol2dPCA(const ArrayList<Matrix>& imageList, 
		   Vector& rowEigenValues, Matrix& rowBasis, 
		   Vector& colEigenValues, Matrix& colBasis, Matrix& mean);
  void Get2dBasisMajor(double rowPart, const Vector& rowEigenValues, 
		       const Matrix& rowBasis, double colPart, 
		       const Vector& colEigenValues, const Matrix& colBasis,
		       Matrix& rowBasisMajor_out, Matrix& colBasisMajor_out);
  void Project2dBasis(const Matrix& A, const Matrix& rowBasis, 
		      const Matrix& colBasis, Matrix& A_out);

}; // namespace

namespace __2D_Factor {
  class Image {
    Matrix m_mX;
    
    ArrayList<double> m_adScale;
    ArrayList<index_t> m_aiRow;
    ArrayList<index_t> m_aiCol;

    void PopulateNN(double row, double col);
  public:
    double Interpolate(double dRow, double dCol);
    Image(const Matrix& X, int window);
  };

  // rectangle in R^2, not necessarily axis parellel
  class BoundingBox {
    double m_dRot; // clockwise rotation 
    double m_dTopRow, m_dTopCol; // top point position
    double m_dLeftArm, m_dRightArm; // lengths of left arm and right arm
  }
};

#endif
