package org.mlpack;

import org.bytedeco.javacpp.annotation.*;

import java.util.Objects;

import org.nd4j.linalg.api.ndarray.INDArray;

@Platform(not = "")
public class MatrixWithInfo {
  private final INDArray matrix;
  private final boolean[] info;

  private static void precondition(boolean value, String format, Object... args) {
    if (!value) {
      throw new IllegalArgumentException(String.format(format, args));
    }
  }

  public MatrixWithInfo(INDArray matrix, boolean[] info) {
    Objects.requireNonNull(matrix);
    Objects.requireNonNull(info);

    precondition(matrix.rank() == 2, "Given n-dimensional array is not a matrix");
    precondition(matrix.dataType() == CLI.FP_TYPE,
        "Matrix has %s data type but %s is expected", matrix.dataType(), CLI.FP_TYPE);
    precondition(info.length == matrix.size(0),
      "Length of 'info' must be equal to the number of rows in the 'matrix'");

    this.matrix = matrix;
    this.info = info;
  }

  public boolean[] getInfo() {
    return info;
  }

  public INDArray getMatrix() {
    return matrix;
  }
}
