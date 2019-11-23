package org.mlpack;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.util.*;

@Platform(include = "java_bindings_test_main.cpp")
public class BindingsTest {
  private static final String THIS_NAME = "Java binding test";

  private static native void mlpackMain();

  public static final class Params {
    private final Map<String, Object> params = new HashMap<>();

    public Params() {
      params.put("string_in", null);
      params.put("int_in", null);
      params.put("double_in", null);
      params.put("flag1", null);
      params.put("flag2", null);
      params.put("matrix_in", null);
      params.put("umatrix_in", null);
      params.put("col_in", null);
      params.put("ucol_in", null);
      params.put("row_in", null);
      params.put("urow_in", null);
      params.put("vector_in", null);
      params.put("str_vector_in", null);
      params.put("umatrix_order_in", null);
      params.put("umatrix_order_out", null);
      //params.put("build_model", null);
      params.put("string_out", null);
      params.put("int_out", null);
      params.put("double_out", null);
      params.put("matrix_out", null);
      params.put("umatrix_out", null);
      params.put("col_out", null);
      params.put("ucol_out", null);
      params.put("row_out", null);
      params.put("urow_out", null);
      params.put("vector_out", null);
      params.put("str_vector_out", null);
      
    }

    public void put(String name, Object value) {
      if (!params.containsKey(name)) {
        throw new IllegalArgumentException(THIS_NAME + " doesn't have " + name + " parameter");
      }

      params.put(name, value);
    }

    private static <T> T tryCast(String name, Object value, Class<T> clazz) {
      try {
        return clazz.cast(value);
      } catch (ClassCastException e) {
        throw new IllegalArgumentException(
            "Parameter " + name + " is not an instance of " + clazz.getName(), e);
      }
    }

    public <T> T get(String name, Class<T> clazz) {
      if (!params.containsKey(name)) {
        throw new IllegalArgumentException(THIS_NAME + " doesn't have " + name + " parameter");
      }

      return tryCast(name, params.get(name), clazz);
    }
  }

  static {
    Loader.load();
  }

  private BindingsTest() {
  }

  private static void checkHasRequiredParameter(Params params, String name) {
    if (params.get(name, Object.class) == null) {
      throw new RuntimeException("Missing required parameter " + name);
    }
  }

  public static void run(Params params) {
    CLI.restoreSettings(THIS_NAME);

    {
      String name = "string_in";
      checkHasRequiredParameter(params, name);
      String value = params.get(name, String.class);
      CLI.setStringParam(name, value);
      CLI.setPassed(name);
    }

    {
      String name = "double_in";
      checkHasRequiredParameter(params, name);
      double value = params.get(name, Double.class);
      CLI.setDoubleParam(name, value);
      CLI.setPassed(name);
    }

    {
      String name = "int_in";
      checkHasRequiredParameter(params, name);
      int value = params.get(name, Integer.class);
      CLI.setIntParam(name, value);
      CLI.setPassed(name);
    }

    {
      String name = "flag1";
      Boolean value = params.get(name, Boolean.class);
      if (value != null) {
        CLI.setBoolParam(name, value);
        CLI.setPassed(name);
      }
    }

    {
      String name = "flag2";
      Boolean value = params.get(name, Boolean.class);
      if (value != null) {
        CLI.setBoolParam(name, value);
        CLI.setPassed(name);
      }
    }

    {
      String name = "matrix_in";
      INDArray value = params.get(name, INDArray.class);
      if (value != null) {
        CLI.setMatParam(name, value);
        CLI.setPassed(name);
      }
    }

    {
      String name = "umatrix_in";
      INDArray value = params.get(name, INDArray.class);
      if (value != null) {
        CLI.setUMatParam(name, value);
        CLI.setPassed(name);
      }
    }

    {
      String name = "umatrix_order_in";
      INDArray value = params.get(name, INDArray.class);
      if (value != null) {
        CLI.setUMatParam(name, value);
        CLI.setPassed(name);
      }
    }

    {
      String name = "col_in";
      INDArray value = params.get(name, INDArray.class);
      if (value != null) {
        CLI.setColParam(name, value);
        CLI.setPassed(name);
      }
    }

    {
      String name = "ucol_in";
      INDArray value = params.get(name, INDArray.class);
      if (value != null) {
        CLI.setUColParam(name, value);
        CLI.setPassed(name);
      }
    }

    {
      String name = "row_in";
      INDArray value = params.get(name, INDArray.class);
      if (value != null) {
        CLI.setRowParam(name, value);
        CLI.setPassed(name);
      }
    }

    {
      String name = "urow_in";
      INDArray value = params.get(name, INDArray.class);
      if (value != null) {
        CLI.setURowParam(name, value);
        CLI.setPassed(name);
      }
    }

    {
      String name = "vector_in";
      Object value = params.get(name, Object.class);
      if (value != null) {
        if (value instanceof List) {
          CLI.setVecIntParam(name, (List) value);
        } else if (value instanceof int[]) {
          CLI.setVecIntParam(name, (int[]) value);
        } else {
          throw new IllegalArgumentException("Parameter " + name + " has invalid type");
        }

        CLI.setPassed(name);
      }
    }

    {
      String name = "str_vector_in";
      Object value = params.get(name, Object.class);
      if (value != null) {
        if (value instanceof List) {
          CLI.setVecStringParam(name, (List) value);
        } else if (value instanceof String[]) {
          CLI.setVecStringParam(name, (String[]) value);
        } else {
          throw new IllegalArgumentException("Parameter " + name + " has invalid type");
        }

        CLI.setPassed(name);
      }
    }

    mlpackMain();

    params.put("string_out", CLI.getStringParam("string_out"));
    params.put("matrix_out", CLI.getMatParam("matrix_out"));
    params.put("double_out", CLI.getDoubleParam("double_out"));
    params.put("int_out", CLI.getIntParam("int_out"));
    params.put("umatrix_out", CLI.getUMatParam("umatrix_out"));
    params.put("col_out", CLI.getColParam("col_out"));
    params.put("ucol_out", CLI.getUColParam("ucol_out"));
    params.put("row_out", CLI.getRowParam("row_out"));
    params.put("urow_out", CLI.getURowParam("urow_out"));
    params.put("vector_out", CLI.getVecIntParam("vector_out"));
    params.put("str_vector_out", CLI.getVecStringParam("str_vector_out"));
    params.put("umatrix_order_out", CLI.getUMatParam("umatrix_order_out"));
  }
}
