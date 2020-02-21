package org.mlpack;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = {"cli_util.hpp", "deleter.hpp"})
@Namespace("mlpack::util")
class CLI {
  private static final char ARMA_ORDER = 'f';
  static final DataType FP_TYPE = DataType.DOUBLE;
  static final DataType UNSIGNED_TYPE = DataType.UINT64;

  private static class ManagedPointer extends Pointer {
    private static class MethodDeallocator extends ManagedPointer implements Deallocator {
      private MethodDeallocator(ManagedPointer p) {
        super(p);
      }

      @Override
      public void deallocate() {
        delete(this);
      }

      @Namespace("::mlpack::util")
      @Name("Delete<unsigned char[]>")
      private static native void delete(Pointer p);
    }

    private ManagedPointer(Pointer p) {
      super(p);
    }

    static Pointer create(Pointer p) {
      ManagedPointer result = new ManagedPointer(p);
      result.deallocator(new MethodDeallocator(result));
      return result;
    }
  }

  static {
    Loader.load();
  }

  private CLI() {
  }

  private static void argumentCheck(boolean value, String format, Object... args) {
    if (!value) {
      throw new IllegalArgumentException(String.format(format, args));
    }
  }

  @Name("RestoreSettings")
  static native void restoreSettings(String name);

  @Name("SetPassed")
  static native void setPassed(String name);

  @Name("EnableVerbose")
  static native void enableVerbose();

  @Name("DisableVerbose")
  static native void disableVerbose();

  // setters

  @Name("SetMatParam<double>")
  private static native void nativeSetMatParam(String name,
      DoublePointer data, long rows, long columns);

  @Name("SetMatParam<size_t>")
  private static native void nativeSetMatParam(String name,
      SizeTPointer data, long rows, long columns);

  static void setMatParam(String name, INDArray mat) {
    argumentCheck(mat.rank() == 2, "Passed argument is not a 2D matrix.");
    argumentCheck(mat.dataType() == FP_TYPE,
        "Matrix data type is %s. %s expected.", mat.dataType(), FP_TYPE);

    if (mat.ordering() != ARMA_ORDER) {
      mat = mat.dup(ARMA_ORDER);
    }

    Pointer data = mat.data().addressPointer();
    int rows = mat.rows();
    int columns = mat.columns();

    nativeSetMatParam(name, new DoublePointer(data), rows, columns);
  }

  static void setUMatParam(String name, INDArray mat) {
    argumentCheck(mat.rank() == 2, "Passed argument is not a 2D matrix.");
    argumentCheck(mat.dataType() == UNSIGNED_TYPE,
        "Matrix data type is %s. %s expected.", mat.dataType(), UNSIGNED_TYPE);

    if (mat.ordering() != ARMA_ORDER) {
      mat = mat.dup(ARMA_ORDER);
    }

    Pointer data = mat.data().addressPointer();
    int rows = mat.rows();
    int columns = mat.columns();

    nativeSetMatParam(name, new SizeTPointer(data), rows, columns);
  }

  @Name("SetParam<int>")
  static native void setIntegerParam(String name, int value);

  @Name("SetParam<bool>")
  static native void setBooleanParam(String name, boolean value);

  @Name("SetParam<double>")
  static native void setDoubleParam(String name, double value);

  @Name("SetParam<std::string>")
  static native void setStringParam(String name, @StdString String value);

  @Name("SetVecElement<int>")
  private static native void setVecElement(String name, int position, int element);

  @Name("SetVecElement<std::string>")
  private static native void setVecElement(String name, int position, String element);

  @Name("SetVecSize<int>")
  private static native void setVecIntSize(String name, int size);

  @Name("SetVecSize<std::string>")
  private static native void setVecStringSize(String name, int size);

  static void setVecIntParam(String name, List<?> data) {
    setVecIntSize(name, data.size());
    for (int i = 0, n = data.size(); i < n; ++i) {
      Object value = data.get(i);
      if (!(value instanceof Integer)) {
        throw new IllegalArgumentException("List element is not an integer");
      }

      setVecElement(name, i, (Integer) value);
    }
  }

  static void setVecStringParam(String name, List<?> data) {
    setVecStringSize(name, data.size());
    for (int i = 0, n = data.size(); i < n; ++i) {
      Object value = data.get(i);
      if (!(value instanceof String)) {
        throw new IllegalArgumentException("List element is not an String");
      }

      setVecElement(name, i, (String) value);
    }
  }

  @Name("SetColParam<double>")
  private static native void nativeSetColParam(String name, DoublePointer data, long size);

  @Name("SetColParam<std::size_t>")
  private static native void nativeSetColParam(String name, SizeTPointer data, long size);

  static void setColParam(String name, INDArray array) {
    argumentCheck(array.isColumnVectorOrScalar(), "Argument is not a column vector");
    argumentCheck(array.dataType() == FP_TYPE,
        "Column data type is %s but %s is expected", array.dataType(), FP_TYPE);

    if (array.ordering() == 'c') {
      array = array.dup();
    }

    Pointer data = array.data().addressPointer();
    long size = array.length();

    nativeSetColParam(name, new DoublePointer(data), size);
  }

  static void setUColParam(String name, INDArray array) {
    argumentCheck(array.isColumnVectorOrScalar(), "Argument is not a column vector");
    argumentCheck(array.dataType() == UNSIGNED_TYPE,
        "Column data type is %s but %s is expected", array.dataType(), UNSIGNED_TYPE);

    if (array.ordering() == 'c') {
      array = array.dup();
    }

    Pointer data = array.data().addressPointer();
    long size = array.length();

    nativeSetColParam(name, new SizeTPointer(data), size);
  }

  @Name("SetRowParam<double>")
  private static native void nativeSetRowParam(String name, DoublePointer data, long size);

  @Name("SetRowParam<std::size_t>")
  private static native void nativeSetRowParam(String name, SizeTPointer data, long size);

  static void setRowParam(String name, INDArray array) {
    argumentCheck(array.isRowVectorOrScalar(), "Argument is not a row vector");
    argumentCheck(array.dataType() == FP_TYPE,
        "Column data type is %s but %s is expected", array.dataType(), FP_TYPE);

    if (array.ordering() == 'f') {
      array = array.dup();
    }

    Pointer data = array.data().addressPointer();
    long size = array.length();

    nativeSetRowParam(name, new DoublePointer(data), size);
  }

  static void setURowParam(String name, INDArray array) {
    argumentCheck(array.isRowVectorOrScalar(), "Argument is not a row vector");
    argumentCheck(array.dataType() == UNSIGNED_TYPE,
        "Column data type is %s but %s is expected", array.dataType(), UNSIGNED_TYPE);

    if (array.ordering() == 'f') {
      array = array.dup();
    }

    Pointer data = array.data().addressPointer();
    long size = array.length();

    nativeSetRowParam(name, new SizeTPointer(data), size);
  }

  @Name("SetMatWithInfoParam")
  private static native void nativeSetMatWithInfoParam(
      String name, DoublePointer data, @Cast("bool*") boolean[] info, long rows, long columns);

  static void setMatWithInfoParam(String name, MatrixWithInfo value) {
    INDArray matrix = value.getMatrix();
    boolean[] info = value.getInfo();

    if (matrix.ordering() != ARMA_ORDER) {
      matrix = matrix.dup(ARMA_ORDER);
    }

    DoublePointer data = new DoublePointer(matrix.data().addressPointer());
    long rows = matrix.rows();
    long columns = matrix.columns();
    nativeSetMatWithInfoParam(name, data, info, rows, columns);
  }

  // getters

  @Name("GetParam<int>")
  static native int getIntegerParam(String name);

  @Name("GetParam<double>")
  static native double getDoubleParam(String name);

  @Name("GetParam<bool>")
  static native boolean getBooleanParam(String name);

  @Name("GetParam<std::string>")
  @StdString
  static native String getStringParam(String name);

  @Name("GetArmaParamData<arma::mat>")
  private static native DoublePointer getMatParamData(String name);

  @Name("GetArmaParamData<arma::Mat<std::size_t>>")
  private static native SizeTPointer getUMatParamData(String name);

  @Name("GetArmaParamRows<arma::mat>")
  private static native long getMatParamRows(String name);

  @Name("GetArmaParamRows<arma::Mat<std::size_t>>")
  private static native long getUMatParamRows(String name);

  @Name("GetArmaParamColumns<arma::mat>")
  private static native long getMatParamColumns(String name);

  @Name("GetArmaParamColumns<arma::Mat<std::size_t>>")
  private static native long getUMatParamColumns(String name);

  @Name("GetArmaParamLength<arma::mat>")
  private static native long getMatParamLength(String name);

  @Name("GetArmaParamLength<arma::Mat<std::size_t>>")
  private static native long getUMatParamLength(String name);

  static INDArray getMatParam(String name) {
    Pointer data = getMatParamData(name);

    if (data == null) {
      return null; // if output parameter wasn't passed
    }

    data = ManagedPointer.create(data);

    long rows = getMatParamRows(name);
    long columns = getMatParamColumns(name);
    long length = getMatParamLength(name);
    DataType type = FP_TYPE;
    data.capacity(length);

    DataBuffer buffer = Nd4j.createBuffer(data, length, type);
    long[] shape = {rows, columns};
    long[] stride = Nd4j.getStrides(shape, ARMA_ORDER);
    long offset = 0;
    char ordering = ARMA_ORDER;

    return Nd4j.create(buffer, shape, stride, offset, ordering, type);
  }

  static INDArray getUMatParam(String name) {
    Pointer data = getUMatParamData(name);

    if (data == null) {
      return null; // if output parameter wasn't passed
    }

    data = ManagedPointer.create(data);

    long rows = getUMatParamRows(name);
    long columns = getUMatParamColumns(name);
    long length = getUMatParamLength(name);
    DataType type = UNSIGNED_TYPE;
    data.capacity(length);

    DataBuffer buffer = Nd4j.createBuffer(data, length, type);
    long[] shape = {rows, columns};
    long[] stride = Nd4j.getStrides(shape, ARMA_ORDER);
    long offset = 0;
    char ordering = ARMA_ORDER;

    return Nd4j.create(buffer, shape, stride, offset, ordering, type);
  }

  @Name("GetVecElement<int>")
  private static native int getVecIntElement(String name, int position);

  @Name("GetVecSize<int>")
  private static native int getVecIntSize(String name);

  @Name("GetVecElement<std::string>")
  @StdString
  private static native String getVecStringElement(String name, int position);

  @Name("GetVecSize<std::string>")
  private static native int getVecStringSize(String name);

  static List<Integer> getVecIntParam(String name) {
    int n = getVecIntSize(name);
    List<Integer> result = new ArrayList<>(n);

    for (int i = 0; i < n; ++i) {
      result.add(getVecIntElement(name, i));
    }

    return result;
  }

  static List<String> getVecStringParam(String name) {
    int n = getVecStringSize(name);
    List<String> result = new ArrayList<>(n);

    for (int i = 0; i < n; ++i) {
      result.add(getVecStringElement(name, i));
    }

    return result;
  }

  @Name("GetArmaParamData<arma::vec>")
  private static native DoublePointer getColParamData(String name);

  @Name("GetArmaParamData<arma::Col<size_t>>")
  private static native SizeTPointer getUColParamData(String name);

  @Name("GetArmaParamLength<arma::vec>")
  private static native long getColParamLength(String name);

  @Name("GetArmaParamLength<arma::Col<std::size_t>>")
  private static native long getUColParamLength(String name);

  static INDArray getColParam(String name) {
    Pointer data = getColParamData(name);

    if (data == null) {
      return null; // if output parameter wasn't passed
    }

    data = ManagedPointer.create(data);

    long length = getColParamLength(name);
    DataType type = FP_TYPE;
    data.capacity(length);

    DataBuffer buffer = Nd4j.createBuffer(data, length, type);
    long[] shape = {length, 1};
    long[] stride = Nd4j.getStrides(shape, ARMA_ORDER);
    long offset = 0;
    char ordering = ARMA_ORDER;

    return Nd4j.create(buffer, shape, stride, offset, ordering, type);
  }

  static INDArray getUColParam(String name) {
    Pointer data = getUColParamData(name);

    if (data == null) {
      return null; // if output parameter wasn't passed
    }

    data = ManagedPointer.create(data);

    long length = getUColParamLength(name);
    DataType type = UNSIGNED_TYPE;
    data.capacity(length);

    DataBuffer buffer = Nd4j.createBuffer(data, length, type);
    long[] shape = {length, 1};
    long[] stride = Nd4j.getStrides(shape, ARMA_ORDER);
    long offset = 0;
    char ordering = ARMA_ORDER;

    return Nd4j.create(buffer, shape, stride, offset, ordering, type);
  }

  @Name("GetArmaParamData<arma::rowvec>")
  private static native DoublePointer getRowParamData(String name);

  @Name("GetArmaParamData<arma::Row<size_t>>")
  private static native SizeTPointer getURowParamData(String name);

  @Name("GetArmaParamLength<arma::rowvec>")
  private static native long getRowParamLength(String name);

  @Name("GetArmaParamLength<arma::Row<std::size_t>>")
  private static native long getURowParamLength(String name);

  static INDArray getRowParam(String name) {
    Pointer data = getRowParamData(name);

    if (data == null) {
      return null; // if output parameter wasn't passed
    }

    data = ManagedPointer.create(data);

    long length = getRowParamLength(name);
    DataType type = FP_TYPE;
    data.capacity(length);

    DataBuffer buffer = Nd4j.createBuffer(data, length, type);
    long[] shape = {1, length};
    long[] stride = Nd4j.getStrides(shape, ARMA_ORDER);
    long offset = 0;
    char ordering = ARMA_ORDER;

    return Nd4j.create(buffer, shape, stride, offset, ordering, type);
  }

  static INDArray getURowParam(String name) {
    Pointer data = getURowParamData(name);

    if (data == null) {
      return null; // if output parameter wasn't passed
    }

    data = ManagedPointer.create(data);

    long length = getURowParamLength(name);
    DataType type = UNSIGNED_TYPE;
    data.capacity(length);

    DataBuffer buffer = Nd4j.createBuffer(data, length, type);
    long[] shape = {1, length};
    long[] stride = Nd4j.getStrides(shape, ARMA_ORDER);
    long offset = 0;
    char ordering = ARMA_ORDER;

    return Nd4j.create(buffer, shape, stride, offset, ordering, type);
  }

  @Name("GetMatWithInfoParamData")
  private static native DoublePointer getMatWithInfoParamData(String name);

  @Name("GetMatWithInfoParamCols")
  private static native long getMatWithInfoParamCols(String name);

  @Name("GetMatWithInfoParamRows")
  private static native long getMatWithInfoParamRows(String name);

  @Name("GetMatWithInfoParamLength")
  private static native long getMatWithInfoParamLength(String name);

  @Name("GetMatWithInfoParamInfo")
  private static native BoolPointer getMatWithInfoParamInfo(String name);

  static MatrixWithInfo getMatWithInfoParam(String name) {
    Pointer data = getMatWithInfoParamData(name);

    if (data == null) {
      return null;
    }

    data = ManagedPointer.create(data);

    long rows = getMatWithInfoParamRows(name);
    long columns = getMatWithInfoParamCols(name);
    long length = getMatWithInfoParamLength(name);
    DataType type = FP_TYPE;
    data.capacity(length);

    DataBuffer buffer = Nd4j.createBuffer(data, length, type);
    long[] shape = {rows, columns};
    long[] stride = Nd4j.getStrides(shape, ARMA_ORDER);
    long offset = 0;
    char ordering = ARMA_ORDER;

    INDArray matrix = Nd4j.create(buffer, shape, stride, offset, ordering, type);
    BoolPointer infoPtr = getMatWithInfoParamInfo(name);
    infoPtr.capacity(rows);

    boolean[] info = new boolean[(int)rows];
    for (int i = 0; i < info.length; ++i) {
      info[i] = infoPtr.get(i);
    }

    ManagedPointer.create(infoPtr);

    return new MatrixWithInfo(matrix, info);
  }
}
