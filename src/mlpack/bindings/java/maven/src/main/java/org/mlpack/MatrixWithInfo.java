package org.mlpack;

import java.util.Objects;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MatrixWithInfo {
  private final INDArray matrix;
  private final boolean[] info;
  private final Order infoOrder;

  public enum Order {
    COLUMN_MAJOR, ROW_MAJOR
  }

  private static void precondition(boolean value, String format, Object... args) {
    if (!value) {
      throw new IllegalArgumentException(String.format(format, args));
    }
  }

  public MatrixWithInfo(INDArray matrix, boolean[] info, Order infoOrder) {
    Objects.requireNonNull(matrix);
    Objects.requireNonNull(info);
    Objects.requireNonNull(infoOrder);

    precondition(matrix.rank() == 2, "Given n-dimensional array is not a matrix");
    precondition(matrix.dataType() == CLI.FP_TYPE, 
        "Matrix has %s data type but %s is expected", matrix.dataType(), CLI.FP_TYPE);

    if (infoOrder == Order.ROW_MAJOR) {
      precondition(info.length == matrix.size(1), 
        "Length of 'info' must be equal to the number of columns in the 'matrix'");
    } else {
      precondition(info.length == matrix.size(0), 
        "Length of 'info' must be equal to the number of rows in the 'matrix'");
    }

    this.matrix = matrix;
    this.info = info;
    this.infoOrder = infoOrder;
  }

  public Order getInfoOrder() {
    return infoOrder;
  }

  public boolean[] getInfo() {
    return info;
  }

  public INDArray getMatrix() {
    return matrix;
  }
}
