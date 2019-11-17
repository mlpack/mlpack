package org.mlpack;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "java_bindings_test_main.cpp")
public class BindingsTest {
    private static final String THIS_NAME = "Java CLI binding test";

    private static native void mlpackMain();

    public static final class Result {
        public INDArray mat;
        public double d;
        public int i;
        public boolean b;
        public String s;
    }

    static {
        Loader.load();
    }

    private BindingsTest() {
    }

    public static Result run(String s, Double d, Integer i, Boolean b, INDArray m) {
        CLI.restoreSettings(THIS_NAME);

        if (s != null) CLI.setParam("string_in", s);
        if (d != null) CLI.setParam("double_in", d);
        if (i != null) CLI.setParam("int_in", i);
        if (m != null) CLI.setMatParam("matrix_in", m, DataType.DOUBLE);
        if (b != null) CLI.setParam("flag_in", b);

        mlpackMain();

        Result result = new Result();
        result.mat = CLI.getMatParam("matrix_out", DataType.DOUBLE);
        result.d = CLI.getDoubleParam("double_out");
        result.i = CLI.getIntParam("int_out");
        result.b = CLI.getBooleanParam("flag_out");
        result.s = CLI.getStringParam("string_out");

        return result;
    }
}
