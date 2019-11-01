package org.mlpack;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "../../cli_util.hpp")
@Namespace("mlpack::util")
class CLI {
  private static final char ARMA_ORDER = 'f';

  private static final class ManagedPointer extends Pointer {
    private static final class MethodDeallocator extends ManagedPointer implements Deallocator {
      private MethodDeallocator(ManagedPointer p) {
        super(p);
      }

      @Override
      public void deallocate() {
        delete(this);
      }

      @Namespace("::mlpack::util")
      @Name("DeleteArray")
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

  @Name("SetMatParam<double>")
  private static native void nativeSetMatParam(String name, 
      DoublePointer data, long rows, long columns);

  @Name("SetMatParam<size_t>")
  private static native void nativeSetMatParam(String name, 
      LongPointer data, long rows, long columns);

  static void setMatParam(String name, INDArray mat, DataType type) {
    argumentCheck(mat.rank() == 2, "Passed argument is not a 2D matrix.");
    argumentCheck(mat.dataType() == type, 
        "Matrix data type is %s. %s expected.", mat.dataType(), type);

    // TODO(Vasniktel): decide the best place to do it: here or in C++
    // So far it seems there is no way to avoid it
    if (mat.ordering() != ARMA_ORDER) {
      mat = mat.dup(ARMA_ORDER);
    }

    Pointer data = mat.data().addressPointer();
    int rows = mat.rows();
    int columns = mat.columns();

    switch (type) {
      case DataType.DOUBLE:
        nativeSetMatParam(name, new DoublePointer(data), rows, columns);
        break;
      case DataType.LONG:
        nativeSetMatParam(name, new LongPointer(data), rows, columns);
        break;
      default:
        throw new UnsupportedOperationException("Matrix elemnt type " + type + " is not supported");
    }
  }

  @Name("SetParam<int>")
  static native void setParam(String name, int value);

  @Name("SetParam<bool>")
  static native void setParam(String name, boolean value);

  @Name("SetParam<double>")
  static native void setParam(String name, double value);

  @Name("SetParam<std::string>")
  static native void setParam(String name, String value);

  @Name("GetMatParamData")
  private static native Pointer getMatParamData(String name);

  @Name("GetMatParamRows")
  private static native long getMatParamRows(String name);

  @Name("GetMatParamColumns")
  private static native long getMatParamColumns(String name);

  @Name("GetMatParamLength")
  private static native long getMatParamLength(String name);

  static INDArray getMatParam(String name, DataType type) {
    Pointer data = ManagedPointer.create(getMatParamData(name));

    long rows = getMatParamRows(name);
    long columns = getMatParamColumns(name);
    long length = getMatParamLength(name);

    DataBuffer buffer = Nd4j.createBuffer(data, length, type);
    long[] shape = {rows, cols};
    long[] stride = Nd4j.getStrides(shape, ARMA_ORDER);
    long offset = 0;
    char ordering = ARMA_ORDER;

    return Nd4j.create(buffer, shape, stride, offset, ordering, type);
  }
}
