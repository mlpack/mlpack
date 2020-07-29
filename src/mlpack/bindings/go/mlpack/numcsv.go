package mlpack

import (
    "encoding/csv"
    "io"
    "os"
    "strconv"
    "net/http"
    "compress/gzip"
    "gonum.org/v1/gonum/mat"
)

// Load() reads all of the numeric records from the CSV.
func Load(filename string) (*mat.Dense, error) {
  var elements int
  var rows int
  var numbers []float64

  // Open the file.
  file, err := os.Open(filename)
  if err != nil {
    return nil, err
  }
  defer file.Close()

  lines, err := csv.NewReader(file).ReadAll()
  if err != nil {
    return nil, err
  }
  
  var str_err error
  for _, args := range lines[0:] {
    for _, arg := range args {
      n, err := strconv.ParseFloat(arg, 64)
      str_err = err 
      if err == nil {
        numbers = append(numbers, n)
        elements = elements + 1
      }
    }
    if str_err == nil {
      rows = rows + 1
    }
  }
  data := numbers[:elements]
  columns := elements/rows
  output := mat.NewDense(rows , columns, data)
  return output, nil
}

// Save() writes all of the records to the CSV.
func Save(filename string, mat *mat.Dense) error {
  // Create the file.
  file, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer file.Close()

  writer := csv.NewWriter(file)
  defer writer.Flush()

  if mat != nil {
    rows, cols := mat.Dims()
    for i := 0; i < rows; i++ {
      var strings []string
      for j := 0; j < cols; j++ {
        s := strconv.FormatFloat(mat.At(i,j), 'e', 16, 64)
        strings = append(strings, s)
      }
      err := writer.Write(strings)
      if err != nil {
        return err
      }
    }
  }
  return nil
}

// UnZip() unzips the given input to the given output file.
func UnZip(input string, output string) error {
    // Create the file.
    out, err := os.Create(output)
    if err != nil {
        return err
    }
    defer out.Close()

    // Open the file.
    in, err := os.Open(input)
    if err != nil {
        return err
    }
    defer in.Close()
    
    // Unzip the data.
    resp, err := gzip.NewReader(in)
    if err != nil {
        return err
    }
    defer resp.Close()

    // Write the body to file.
    _, err = io.Copy(out, resp)
    if err != nil {
        return err
    }
    return nil
}

// DownloadFile() downloads the file from the given url and
// save it to the given filename.
func DownloadFile (url string, filename string) error {
    // Create the file.
    out, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer out.Close()

    // Get the data.
    resp, err := http.Get(url)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    // Write the body to file.
    _, err = io.Copy(out, resp.Body)
    if err != nil {
        return err
    }

    return nil
}
