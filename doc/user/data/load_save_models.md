## mlpack objects

Machine learning models (aka mlpack objects) can be saved with
`data::Save()` and loaded with `data::Load()`.  Serialization is performed
using the [cereal](https://uscilab.github.io/cereal/) serialization toolkit.

 - `data::Load(filename, object, opts)`
 - `data::Save(filename, object, opts)`

   * Load/save `object` to/from `filename`.

   * If `opts.Fatal()` is `true`, a `std::runtime_error` will be thrown in the
     event of load or save failure.

   * The format is autodetected based on extension (`.bin`, `.json`, or `.xml`),
     but can be manually specified:
     - `opts.Format() = FileType::BIN`: binary blob (smallest and fastest).
       No checks; assumes all data is correct.
     - `opts.Format() = FileType::JSON`: JSON.
     - `opts.Format() = FileType::XML`: XML (largest and slowest).

   * Returns a `bool` indicating the success of the operation.

***Note:*** when loading an object that was saved as a binary blob, the C++ type
of the object must be ***exactly the same*** (including template parameters) as
the type used to save the object.  If not, undefined behavior will occur---most
likely a crash.

---

These functions can be also used with the following siganture for simplicity:

``` 
    // To load as JSON format, and throw an exception if fatal.
    mlpack::data::Load(filename, object, Fatal + JSON);

    // To save as binary format, and show a warning if fatal.
    // Note MLPACK_PRINT_WARN need to be defined when configuring cmake to see
    // the warning.
    mlpack::data::Save(filename, object, NoFatal + BIN);
```

---

Simple example: create a `math::Range` object, then save and load it.

```c++
mlpack::math::Range r(3.0, 6.0);

// How we can use DataOptions with loading / saving objects.
data::DataOptions opts;
opts.Fatal() = true;
opts.Format() = FileType::BIN;

// Save the Range to 'range.bin', using the name "range".
mlpack::data::Save("range.bin", r, opts);

// Load the range into a new object.
mlpack::math::Range r2;
mlpack::data::Load("range.bin", r2, BIN + Fatal);

std::cout << "Loaded range: [" << r2.Lo() << ", " << r2.Hi() << "]."
    << std::endl;

// Modify and save the range as JSON.
r2.Lo() = 4.0;
mlpack::data::Save("range.json", r2, JSON + Fatal);

// Now 'range.json' will contain the following:
//
// {
//     "range": {
//         "cereal_class_version": 0,
//         "hi": 6.0,
//         "lo": 4.0
//     }
// }
```

---

