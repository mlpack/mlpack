package main

import (
	"mlpack/bindings/go/mlpack"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestRunBindingNoFlag(t *testing.T) {
	t.Log("Test that if we forget the mandatory flag, we should get wrong",
        "results.")
  param := mlpack.InitializeTestGoBinding()
  d := 4.0
  i := 12
  s := "hello"
  _, DoubleOut, IntOut, _, _, _, _, _, _, StringOut, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  if DoubleOut == 5.0 {
    t.Errorf("Error. Wrong DoubleOut value.")
  }
  if IntOut == 13 {
    t.Errorf("Error. Wrong IntOut value.")
  }
  if StringOut == "hello2" {
    t.Errorf("Error. Wrong StringOut value.")
  }
}

func TestRunBindingCorrectly(t *testing.T) {
	t.Log("Test that when we run the binding correctly (with correct",
        " input parameters), we get the expected output.")
  param := mlpack.InitializeTestGoBinding()
  param.Flag1 = true
  d := 4.0
  i := 12
  s := "hello"
  _, DoubleOut, IntOut, _, _, _, _, _, _, StringOut, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  if DoubleOut != 5.0 {
    t.Errorf("Error. Wrong DoubleOut value.")
  }
  if IntOut != 13 {
    t.Errorf("Error. Wrong IntOut value.")
  }
  if StringOut != "hello2" {
    t.Errorf("Error. Wrong StringOut value.")
  }
}

func TestRunBindingWrongString(t *testing.T) {
  t.Log("Test that if we give the wrong string, we should get wrong results.")
  param := mlpack.InitializeTestGoBinding()
  param.Flag1 = true
  d := 4.0
  i := 12
  s := "goodbye"
  _, _, _, _, _, _, _, _, _, StringOut, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  if StringOut == "hello2" {
    t.Errorf("Error. Wrong StringOut value.")
  }
}

func TestRunBindingWrongInt(t *testing.T) {
  t.Log("Test that if we give the wrong int, we should get wrong results.")
  param := mlpack.InitializeTestGoBinding()
  param.Flag1 = true
  d := 4.0
  i := 15
  s := "hello"
  _, _, IntOut, _, _, _, _, _, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  if IntOut == 13 {
    t.Errorf("Error. Wrong IntOut value.")
  }

}

func TestRunBindingWrongDouble(t *testing.T) {
  t.Log("Test that if we give the wrong double, we should get wrong results.")
  param := mlpack.InitializeTestGoBinding()
  param.Flag1 = true
  d := 2.0
  i := 12
  s := "hello"
  _, DoubleOut, _, _, _, _, _, _, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  if DoubleOut == 5.0 {
    t.Errorf("Error. Wrong DoubleOut value.")
  }
}

func TestRunBadFlag(t *testing.T) {
  t.Log("Testing that if we give a second flag, it should fail.")
  param := mlpack.InitializeTestGoBinding()
  param.Flag1 = true
  param.Flag2 = true
  d := 2.0
  i := 12
  s := "hello"
  _, DoubleOut, IntOut, _, _, _, _, _, _, StringOut, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  if DoubleOut == 5.0 {
    t.Errorf("Error. Wrong DoubleOut value.")
  }
  if IntOut == 13 {
    t.Errorf("Error. Wrong IntOut value.")
  }
  if StringOut == "hello2" {
    t.Errorf("Error. Wrong StringOut value.")
  }
}

func TestGonumMatrix(t *testing.T) {
  t.Log("Test that the matrix we get back should be the matrix we pass in",
        "with the third dimension doubled and the fifth forgotten.")
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

  param := mlpack.InitializeTestGoBinding()
  param.MatrixIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, MatrixOut, _, _, _, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  rows, cols := MatrixOut.Dims()
  if rows != 3 || cols != 4 {
    panic("error shape")
  }

  var z mat.Dense
  z.Sub(MatrixOut, y)
  for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
      if val := z.At(i, j); val != 0 {
        t.Errorf("Error. Value at [i,j] : %v", val)
      }
    }
  }
}

func TestGonumUMatrix(t *testing.T) {
  t.Log("Test that the umatrix we get back should be the umatrix we pass",
        "in with the third dimension doubled and the fifth forgotten.")
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

  param := mlpack.InitializeTestGoBinding()
  param.UmatrixIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, _, _, _,  _, UmatrixOut, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  rows, cols := UmatrixOut.Dims()
  if rows != 3 || cols != 4 {
    panic("error shape")
  }

  var z mat.Dense
  z.Sub(UmatrixOut, y)
  for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
      if val := z.At(i, j); val != 0 {
        t.Errorf("Error. Value at [i,j] : %v", val)
      }
    }
  }
}
func TestGonumTransposeRow(t *testing.T) {
  t.Log("Test a column vector input parameter.")
  x := mat.NewDense(1, 9, []float64{
    1, 2, 3, 4, 5, 6, 7, 8, 9,
  })

  param := mlpack.InitializeTestGoBinding()
  param.RowIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, RowOut, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  rows, _ := RowOut.Dims()
  if rows != 9 {
    t.Errorf("Error. Wrong shape.")
  }
  for i := 0; i < rows; i++ {
    if RowOut.At(i, 0) != x.At(0, i)*2 {
      val := RowOut.At(i, 0)
      expected := x.At(0, i) * 2
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestGonumTransposeURow(t *testing.T) {
  t.Log("Test a column vector input parameter.")
  x := mat.NewDense(1, 9, []float64{
    1, 2, 3, 4, 5, 6, 7, 8, 9,
  })

  param := mlpack.InitializeTestGoBinding()
  param.UrowIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, _, _, _, _, _, UrowOut, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  urows, _ := UrowOut.Dims()
  if urows != 9 {
    t.Errorf("Error. Wrong shape.")
  }
  for i := 0; i < urows; i++ {
    if UrowOut.At(i, 0) != x.At(0, i)*2 {
      val := UrowOut.At(i, 0)
      expected := x.At(0, i) * 2
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestGonumTransposeCol(t *testing.T) {
  t.Log("Test a row vector input parameter.")
  x := mat.NewDense(9, 1,  []float64{
    1, 2, 3, 4, 5, 6, 7, 8, 9,
  })

  param := mlpack.InitializeTestGoBinding()
  param.ColIn = x
  d := 4.0
  i := 12
  s := "hello"
  ColOut, _, _, _, _, _, _, _, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  _, cols := ColOut.Dims()
  if cols != 9 {
    t.Errorf("Error. Wrong shape.")
  }
  for i := 0; i < cols; i++ {
    if ColOut.At(0, i) != x.At(i, 0)*2 {
      val := ColOut.At(0, i)
      expected := x.At(i, 0) * 2
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestGonumTransposeUCol(t *testing.T) {
  t.Log("Test a row vector input parameter.")
  x := mat.NewDense(9, 1, []float64{
    1, 2, 3, 4, 5, 6, 7, 8, 9,
  })

  param := mlpack.InitializeTestGoBinding()
  param.UcolIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, _, _, _, UcolOut, _, _, _:=
       mlpack.TestGoBinding(d, i, s, param)

  _, ucols := UcolOut.Dims()
  if ucols != 9 {
    t.Errorf("Error. Wrong shape.")
  }
  for i := 0; i < ucols; i++ {
    if UcolOut.At(0, i) != x.At(i, 0)*2 {
      val := UcolOut.At(0, i)
      expected := x.At(i, 0) * 2
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestGonumRow(t *testing.T) {
  t.Log("Test a row vector input parameter.")
  x := mat.NewDense(9, 1,  []float64{
    1, 2, 3, 4, 5, 6, 7, 8, 9,
  })

  param := mlpack.InitializeTestGoBinding()
  param.RowIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, RowOut, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  rows, _ := RowOut.Dims()
  if rows != 9 {
    t.Errorf("Error. Wrong shape.")
  }
  for i := 0; i < rows; i++ {
    if RowOut.At(i, 0) != x.At(i, 0)*2 {
      val := RowOut.At(i, 0)
      expected := x.At(i, 0) * 2
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestGonumURow(t *testing.T) {
  t.Log("Test a row vector input parameter.")
  x := mat.NewDense(9, 1, []float64{
    1, 2, 3, 4, 5, 6, 7, 8, 9,
  })

  param := mlpack.InitializeTestGoBinding()
  param.UrowIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, _, _, _, _, _, UrowOut, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  urows, _ := UrowOut.Dims()
  if urows != 9 {
    t.Errorf("Error. Wrong shape.")
  }
  for i := 0; i < urows; i++ {
    if UrowOut.At(i, 0) != x.At(i, 0)*2 {
      val := UrowOut.At(i, 0)
      expected := x.At(i, 0) * 2
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestGonumCol(t *testing.T) {
  t.Log("Test a column vector input parameter.")
  x := mat.NewDense(1, 9, []float64{
    1, 2, 3, 4, 5, 6, 7, 8, 9,
  })

  param := mlpack.InitializeTestGoBinding()
  param.ColIn = x
  d := 4.0
  i := 12
  s := "hello"
  ColOut, _, _, _, _, _, _, _, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  _, cols := ColOut.Dims()
  if cols != 9 {
    t.Errorf("Error. Wrong shape.")
  }
  for i := 0; i < cols; i++ {
    if ColOut.At(0, i) != x.At(0, i)*2 {
      val := ColOut.At(0, i)
      expected := x.At(0, i) * 2
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestGonumUCol(t *testing.T) {
  t.Log("Test a column vector input parameter.")
  x := mat.NewDense(1, 9, []float64{
    1, 2, 3, 4, 5, 6, 7, 8, 9,
  })

  param := mlpack.InitializeTestGoBinding()
  param.UcolIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, _, _, _, UcolOut, _, _, _:=
       mlpack.TestGoBinding(d, i, s, param)

  _, ucols := UcolOut.Dims()
  if ucols != 9 {
    t.Errorf("Error. Wrong shape.")
  }
  for i := 0; i < ucols; i++ {
    if UcolOut.At(0, i) != x.At(0, i)*2 {
      val := UcolOut.At(0, i)
      expected := x.At(0, i) * 2
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestGonumWrongRow(t *testing.T) {
  t.Log("Test a wrong shape row vector input parameter.")
  // A defer statement defers the execution of a function until
  // the surrounding function returns.
  defer func() {
    if r := recover(); r == nil {
      t.Errorf("The code did not panic")
    }
  }()
  x := mat.NewDense(3, 5, []float64{
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
  })

  param := mlpack.InitializeTestGoBinding()
  param.RowIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, RowOut, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  _, err := RowOut.Dims()
  if err == 1 {
    t.Errorf("Error. Working.")
  }
}

func TestGonumWrongCol(t *testing.T) {
  t.Log("Test a wrong shape column vector input parameter.")
  // A defer statement defers the execution of a function until
  // the surrounding function returns.
  defer func() {
    if r := recover(); r == nil {
      t.Errorf("The code did not panic")
    }
  }()
  x := mat.NewDense(3, 5, []float64{
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
  })

  param := mlpack.InitializeTestGoBinding()
  param.RowIn = x
  d := 4.0
  i := 12
  s := "hello"
  ColOut, _, _, _, _, _, _, _, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  err, _ := ColOut.Dims()
  if err == 1 {
    t.Errorf("Error. Working.")
  }
}

func TestRunIntVector(t *testing.T) {
  t.Log("Test a int vector input paramter.")
  param := mlpack.InitializeTestGoBinding()
  x := []int{
    1, 2, 3, 4, 5, 6,
  }

  param.VectorIn = x
  d := 2.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, _, _, _, _, _, _, VectorOut :=
      mlpack.TestGoBinding(d, i, s, param)

  length := len(VectorOut)
  if length != 5 {
    t.Errorf("Error. Wrong Length.")
  }
  for i := 0; i < length; i++ {
    if x[i] != VectorOut[i]{
      val := VectorOut[i]
      expected := x[i]
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestRunStringVector(t *testing.T) {
  t.Log("Test a string vector input paramter.")
  param := mlpack.InitializeTestGoBinding()
  x := []string{
    "1", "2", "3", "4", "5", "6",
  }

  param.StrVectorIn = x
  d := 2.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, _, _, StrVectorOut, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  length := len(StrVectorOut)
  if length != 5 {
    t.Errorf("Error. Wrong Length.")
  }
  for i := 0; i < length; i++ {
    if x[i] != StrVectorOut[i] {
      val := StrVectorOut[i]
      expected := x[i]
      t.Errorf("Error. Value at [i] : %v. Expected value : %v",
               val, expected)
    }
  }
}

func TestGonumMatrixWithInfo(t *testing.T) {
  t.Log("Test that the matrix_withInfo we get back should be the ",
        "matrix_withInfo we pass in with double the element of the matrix .")

  x := mlpack.DataAndInfo()
  x.Categoricals = []bool{
    false, false, false, false, false,
  }

  x.Data = mat.NewDense(3, 5, []float64{
           1, 2, 3, 4, 5,
           6, 7, 8, 9, 10,
           11, 12, 13, 14, 15,
  })

  param := mlpack.InitializeTestGoBinding()
  param.MatrixAndInfoIn = x
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, MatrixAndInfoOut, _, _, _, _, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  rows, cols := MatrixAndInfoOut.Dims()

  if rows != 3 || cols != 5 {
    t.Errorf("Error. Wrong shape. %v, %v", rows, cols)
  }
  for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
      if x.Data.At(i, j)*2 != MatrixAndInfoOut.At(i, j) {
        val := MatrixAndInfoOut.At(i, j)
        expected := x.Data.At(i, j)*2
        t.Errorf("Error. Value at [%v,%v] : %v. Expected value : %v",
                 i, j, val, expected)
      }
    }
  }
}

func TestModel(t *testing.T) {
  t.Log("First create a GaussianKernel object, then send it back and",
        "make sure we get the right double value.")

  param := mlpack.InitializeTestGoBinding()
  param.BuildModel = true
  d := 4.0
  i := 12
  s := "hello"
  _, _, _, _, _, _, ModelOut, _, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param)

  param2 := mlpack.InitializeTestGoBinding()
  param2.ModelIn = &ModelOut
  _, _, _, _, _, ModelBwOut, _, _, _, _, _, _, _, _ :=
      mlpack.TestGoBinding(d, i, s, param2)

  if ModelBwOut != 20.0 {
    t.Errorf("Error. Wrong model.")
  }
}
