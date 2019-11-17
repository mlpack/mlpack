package org.mlpack;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.mlpack.BindingsTest;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

public class BindingsJUnitTest {
    @Test
    public void returnsCorrectMatrix() {
        double[][] data = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        INDArray arr = Nd4j.create(data);
        BindingsTest.Result result = BindingsTest.run(null, null, null, null, arr);

        assertEquals(result.mat, arr.addi(4));
    }

    @Test
    public void returnsCorrectString() {
        String data = "test string";

        BindingsTest.Result result = BindingsTest.run(data, null, null, null, null);

        assertEquals(result.s, data + "_fixed");
    }

    @Test
    public void returnsCorrectDouble() {
        Double data = 45.3;

        BindingsTest.Result result = BindingsTest.run(null, data, null, null, null);

        assertEquals(result.d, data + 4.5, 0.0001);
    }

    @Test
    public void returnsCorrectInt() {
        Integer data = 42;

        BindingsTest.Result result = BindingsTest.run(null, null, data, null, null);

        assertEquals(result.i, data + 4);
    }

    @Test
    public void returnsCorrectBool() {
        Boolean data = true;

        BindingsTest.Result result = BindingsTest.run(null, null, null, data, null);

        assertEquals(result.b, !data);
    }
}
