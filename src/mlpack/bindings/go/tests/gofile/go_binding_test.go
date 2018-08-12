package main

import (
	"mlpack/build/src/mlpack/bindings/go/mlpack"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestRunBindingNoFlag(t *testing.T) {
	t.Log("Test that when we run the binding correctly (with correct input parameters), we get the expected output.")
	param := mlpack.InitializeTest_go_binding()
	param.Copy_all_inputs = true
	d := 4.0
	i := 12
	s := "hello"
	_, double_out, int_out, _, _, _, _, _, string_out, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	if double_out == 5.0 {
		t.Errorf("Error. Wrong double_out value.")
	}
	if int_out == 13 {
		t.Errorf("Error. Wrong int_out value.")
	}
	if string_out == "hello2" {
		t.Errorf("Error. Wrong string_out value.")
	}
}

func TestRunBindingCorrectly(t *testing.T) {
	t.Log("Test that if we forget the mandatory flag, we should get wrong results.")
	param := mlpack.InitializeTest_go_binding()
	param.Copy_all_inputs = true
	param.Flag1 = true
	d := 4.0
	i := 12
	s := "hello"
	_, double_out, int_out, _, _, _, _, _, string_out, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	if double_out != 5.0 {
		t.Errorf("Error. Wrong double_out value.")
	}
	if int_out != 13 {
		t.Errorf("Error. Wrong int_out value.")
	}
	if string_out != "hello2" {
		t.Errorf("Error. Wrong string_out value.")
	}
}

func TestRunBindingWrongString(t *testing.T) {
	t.Log("Test that if we give the wrong string, we should get wrong results.")
	param := mlpack.InitializeTest_go_binding()
	param.Flag1 = true
	d := 4.0
	i := 12
	s := "goodbye"
	_, _, _, _, _, _, _, _, string_out, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	if string_out == "hello2" {
		t.Errorf("Error. Wrong string_out value.")
	}
}

func TestRunBindingWrongInt(t *testing.T) {
	t.Log("Test that if we give the wrong int, we should get wrong results.")
	param := mlpack.InitializeTest_go_binding()
	param.Flag1 = true
	d := 4.0
	i := 15
	s := "hello"
	_, _, int_out, _, _, _, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	if int_out == 13 {
		t.Errorf("Error. Wrong int_out value.")
	}

}

func TestRunBindingWrongDouble(t *testing.T) {
	t.Log("Test that if we give the wrong double, we should get wrong results.")
	param := mlpack.InitializeTest_go_binding()
	param.Flag1 = true
	d := 2.0
	i := 12
	s := "hello"
	_, double_out, _, _, _, _, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	if double_out == 5.0 {
		t.Errorf("Error. Wrong double_out value.")
	}
}

func TestRunBadFlag(t *testing.T) {
	t.Log("Testing that if we give a second flag, it should fail.")
	param := mlpack.InitializeTest_go_binding()
	param.Flag1 = true
	param.Flag2 = true
	d := 2.0
	i := 12
	s := "hello"
	_, double_out, int_out, _, _, _, _, _, string_out, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	if double_out == 5.0 {
		t.Errorf("Error. Wrong double_out value.")
	}
	if int_out == 13 {
		t.Errorf("Error. Wrong int_out value.")
	}
	if string_out == "hello2" {
		t.Errorf("Error. Wrong string_out value.")
	}
}

func TestGonumMatrix(t *testing.T) {
	t.Log("Test that the matrix we get back should be the matrix we pass in with the third dimension doubled and the fifth forgotten.")
	x := mat.NewDense(3, 5, []float64{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
	})

	y := mat.NewDense(3, 4, []float64{
		1, 2, 6, 4,
		6, 7, 16, 9,
		11, 12, 26, 14,
	})

	param := mlpack.InitializeTest_go_binding()
	param.Copy_all_inputs = true
	param.Matrix_in = x
	d := 4.0
	i := 12
	s := "hello"
	_, _, _, matrix_out, _, _, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	rows, cols := matrix_out.Dims()
	if rows != 3 || cols != 4 {
		panic("error shape")
	}

	var z mat.Dense
	z.Sub(matrix_out, y)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if val := z.At(i, j); val != 0 {
				t.Errorf("Error. Value at [i,j] : %v", val)
			}
		}
	}
}

func TestGonumMatrixForceCopy(t *testing.T) {
	t.Log("Test that the matrix we get back should be the matrix we pass in with the third dimension doubled and the fifth forgotten.")
	x := mat.NewDense(3, 5, []float64{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
	})

	param := mlpack.InitializeTest_go_binding()
	param.Matrix_in = x
	d := 4.0
	i := 12
	s := "hello"
	_, _, _, matrix_out, _, _, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	rows, cols := matrix_out.Dims()

	if rows != 3 || cols != 4 {
		t.Errorf("Error. Wrong shape.")
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < 2; j++ {
			if x.At(i, j) != matrix_out.At(i, j) {
				val := matrix_out.At(i, j)
				expected := x.At(i, j)
				t.Errorf("Error. Value at [i,j] : %v. Expected value : %v", val, expected)
			}
		}
	}
	for i := 0; i < rows; i++ {
		if x.At(i, 3) != matrix_out.At(i, 3) {
			val := matrix_out.At(i, 3)
			expected := x.At(i, 3)
			t.Errorf("Error. Value at [i,j] : %v. Expected value : %v", val, expected)
		}
	}

	for i := 0; i < rows; i++ {
		if x.At(i, 2)*2 != matrix_out.At(i, 2) {
			val := matrix_out.At(i, 2)
			expected := x.At(i, 2) * 2
			t.Errorf("Error. Value at [i,j] : %v. Expected value : %v", val, expected)
		}
	}
}

func TestGonumRow(t *testing.T) {
	t.Log("Test a row vector input parameter.")
	x := mat.NewVecDense(9, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
	})
	y := mat.VecDenseCopyOf(x)

	param := mlpack.InitializeTest_go_binding()
	param.Row_in = y
	d := 4.0
	i := 12
	s := "hello"
	_, _, _, _, _, _, row_out, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	rows, _ := row_out.Dims()
	if rows != 9 {
		t.Errorf("Error. Wrong shape.")
	}
	for i := 0; i < rows; i++ {
		if row_out.AtVec(i) != x.AtVec(i)*2 {
			val := row_out.AtVec(i)
			expected := x.AtVec(i) * 2
			t.Errorf("Error. Value at [i] : %v. Expected value : %v", val, expected)
		}
	}
}

func TestGonumRowForceCopy(t *testing.T) {
	t.Log("Test a row vector input parameter.")
	x := mat.NewVecDense(9, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
	})

	param := mlpack.InitializeTest_go_binding()
	param.Copy_all_inputs = true
	param.Row_in = x
	d := 4.0
	i := 12
	s := "hello"
	_, _, _, _, _, _, row_out, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	rows, _ := row_out.Dims()
	if rows != 9 {
		t.Errorf("Error. Wrong shape.")
	}
	for i := 0; i < rows; i++ {
		if row_out.AtVec(i) != x.AtVec(i)*2 {
			val := row_out.AtVec(i)
			expected := x.AtVec(i) * 2
			t.Errorf("Error. Value at [i] : %v. Expected value : %v", val, expected)
		}
	}
}

func TestGonumCol(t *testing.T) {
	t.Log("Test a column vector input parameter.")
	x := mat.NewVecDense(9, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
	})
	y := mat.VecDenseCopyOf(x)

	param := mlpack.InitializeTest_go_binding()
	param.Col_in = y
	d := 4.0
	i := 12
	s := "hello"
	col_out, _, _, _, _, _, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	cols, _ := col_out.Dims()
	if cols != 9 {
		t.Errorf("Error. Wrong shape.")
	}
	for i := 0; i < cols; i++ {
		if col_out.AtVec(i) != x.AtVec(i)*2 {
			val := col_out.AtVec(i)
			expected := x.AtVec(i) * 2
			t.Errorf("Error. Value at [i] : %v. Expected value : %v", val, expected)
		}
	}
}

func TestGonumColForceCopy(t *testing.T) {
	t.Log("Test a column vector input parameter.")
	x := mat.NewVecDense(9, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
	})

	param := mlpack.InitializeTest_go_binding()
	param.Copy_all_inputs = true
	param.Col_in = x
	d := 4.0
	i := 12
	s := "hello"
	col_out, _, _, _, _, _, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	cols, _ := col_out.Dims()
	if cols != 9 {
		t.Errorf("Error. Wrong shape.")
	}
	for i := 0; i < cols; i++ {
		if col_out.AtVec(i) != x.AtVec(i)*2 {
			val := col_out.AtVec(i)
			expected := x.AtVec(i) * 2
			t.Errorf("Error. Value at [i] : %v. Expected value : %v", val, expected)
		}
	}
}

func TestModel(t *testing.T) {
	t.Log("First create a GaussianKernel object, then send it back and make sure we get the right double value.")

	param := mlpack.InitializeTest_go_binding()
	param.Build_model = true
	d := 4.0
	i := 12
	s := "hello"
	_, _, _, _, _, model_out, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	param2 := mlpack.InitializeTest_go_binding()
	param2.Model_in = &model_out
	_, _, _, _, model_bw_out, _, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param2)

	if model_bw_out != 20.0 {
		t.Errorf("Error. Wrong model.")
	}
}

func TestModelForceCopy(t *testing.T) {
	t.Log("First create a GaussianKernel object, then send it back and make sure we get the right double value.")

	param := mlpack.InitializeTest_go_binding()
	param.Build_model = true
	d := 4.0
	i := 12
	s := "hello"
	_, _, _, _, _, model_out, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param)

	param2 := mlpack.InitializeTest_go_binding()
	param2.Model_in = &model_out
	param2.Copy_all_inputs = true
	_, _, _, _, model_bw_out, _, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param2)

	param3 := mlpack.InitializeTest_go_binding()
	param3.Model_in = &model_out
	_, _, _, _, model_bw_out2, _, _, _, _, _, _, _, _ := mlpack.Test_go_binding(d, i, s, param3)

	if model_bw_out != 20.0 {
		t.Errorf("Error. model_bw_out value: %v. model_bw_out expected value: 20.0.", model_bw_out)
	}
	if model_bw_out2 != 20.0 {
		t.Errorf("Error. model_bw_out2 value: %v. model_bw_out2 expected value: 20.0.", model_bw_out2)
	}
}
