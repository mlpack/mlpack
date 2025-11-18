## Data Options

It is a generic class that allows the user to specify the
`data::Load()` and `data::Save()` options when loading and saving dataset
files. mlpack has an identical data load API in whether we
are trying to load an image or a csv file. However, we need to specify the
relevant options for each case. Currently mlpack supports the following:

1. `data::DataOptions`: provide settings if load is fatal or not, and allow
   specify the Format of the File we are trying to load.
2. `data::MatrixOptions` provide settings to transpose matrix when loading /
   saving. Inherits directly `DataOptions`
2. `data::TextOptions`: provide settings related to time series data. Inherits
   directly `MatrixOptions`
3. `data::ImageOptions`: provide setttings when loading images. (e,g., Height,
   Width, etc). Inherits directly `DataOptions`

The settings related to these classes are simplfied to easy interaction. This
is done by using the `+` operator between these settings when loading /
saving. More details are provided below with examples how to use
`data::Load / data::Save` functions.


|-------------------------------------------------------------------------------------------------------|
| Operator | Function | Type  | Comment | 
|-------------------------------------------------------------------------------------------------------|
| `Fatal`  | `.Fatal() = true` | bool | A  `std::runtime_error` will be thrown on failure.              |
|-------------------------------------------------------------------------------------------------------|
| `NoFatal`  | `.Fatal() = false` | bool | A false will be returned on failure.A warning will be printed if the user enabled `MLPACK_PRINT_WARN`|
|-------------------------------------------------------------------------------------------------------|
| `CSV` | `.Format() = FileType::CSVASCII` | enum | (autodetect extensions `.csv`, `.tsv`): CSV format  |
|       |                                  |      | with no header.  If loading a sparse matrix and the |
|       |                                  |      | CSV has three columns, the data is interpreted as a |
|       |                                  |      | [coordinate list](https://arma.sourceforge.net/docs.html#save_load_mat). |
|-----------------------------------------------------------------------------------------------------|
| `PGM` | `.Format() = FileType::PGMBinary` | enum | (autodetect extension `.pgm`): PGM image format. |
|-----------------------------------------------------------------------------------------------------|
| `PPM` | `.Format() = FileType::PPMBinary` | enum | (autodetect extension `.ppm`):PPM image format.  |
|-----------------------------------------------------------------------------------------------------|
| `HDF5` | `.Format() = FileType::HDF5Binary` | enum | (autodetect extensions `.h5`, `.hdf5`, `.hdf`,
|        |                                    |      |  `.he5`): [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) |
|        |                                    |      |   binary format; only available if Armadillo is configured with |
|        |                                    |      | [HDF5 support](https://arma.sourceforge.net/docs.html#config_hpp). |
|--------------------------------------------------------------------------------------------------------------|
| `ArmaAscii` | `.Format() = FileType::ArmaASCII` | enum | (autodetect extension `.txt`): space-separated |
|             |                                   |      | values as saved by Armadillo with the          |
|             |                                   |      | [`arma_ascii`](https://arma.sourceforge.net/docs.html#save_load_mat) |
|             |                                   |      | format. |
|--------------------------------------------------------------------------------------------------------------|
| `ArmaBin` | `.Format() = FileType::ArmaBinary` | enum |  (autodetect extension `.bin`): Armadillo's     |
|           |                                    |      |  efficient binary matrix format                 |
|           |                                    |      |  ([`arma_binary`](https://arma.sourceforge.net/docs.html#save_load_mat)). |
|--------------------------------------------------------------------------------------------------------------|
| `RawAscii` | `.Format() = FileType::RawASCII` | enum | (autodetect extensions `.csv`, `.txt`):          |
|            |                                  |      | space-separated values or tab-separated values (TSV) |
|            |                                  |      | with no header.                                  |
|--------------------------------------------------------------------------------------------------------------------|
| `BinAscii` | `.Format() = FileType::RawBinary` | enum | (autodetect extension `.bin`): packed binary data     | 
|            |                                   |      | with no header and no size information; data will be  |
|            |                                   |      | loaded as a single column vector _(not recommended)_. |
|--------------------------------------------------------------------------------------------------------------------|
| `CoordAscii` | `.Format() = FileType::CoordAscii` | enum | (autodetect extensions `.txt`, `.tsv`; must be loading a sparse |
|              |                                    |      | matrix type): coordinate list format for sparse data (see       |
|              |                                    |      | [`coord_ascii`](https://arma.sourceforge.net/docs.html#save_load_mat)). |
|--------------------------------------------------------------------------------------------------------------------|
| `ARFF` | `.Format() = FileType::ARFFAscii` | enum | (autodetect extensions `.arff`) ARFF filetype. Used specifically to load mixed categorical dataset. |
|        |                                   |      | [ARFF](https://ml.cms.waikato.ac.nz/weka/arff.html).|
|------------------------------------------------------------------------------------------------------------|
| `AutoDetect` | `.Format() = FileType::AutoDetect` | enum | auto-detects the format as one of the  |
|              |                                    |      | formats above using the extension of the |
|              |                                    |      | filename and inspecting the file contents.|
|----------------------------------------------------------------------------------------------------------------|

***Notes:***

   - ASCII formats (`CSVAscii`, `RawAscii`, `ArmaAscii`) are human-readable but
     large; to reduce dataset size, consider a binary format such as
      `ArmaBinary` or `HDF5`.
   - Sparse data (`arma::sp_mat`, `arma::sp_fmat`, etc.) must be saved in a
     binary format (`ArmaBinary` or `HDF5`) or as a coordinate list
     (`CoordAscii`).

### [mlpack objects](#mlpack-objects)

By default, load/save format for mlpack objects is autodetected. However, 
if necessary you can specify the format of the serialization of the objects (ml
models). Currently mlpack supports three serialization formats: Binary, JSON,
and XML. Those can be specified as follows:

|-------------------------------------------------------------------------------|
| operator |  Function | Type  | Comment                                        |
|-------------------------------------------------------------------------------|
| `JSON` | `.Format() = FileType::JSON` | enum | (autodetect extension `.json`) |
|-------------------------------------------------------------------------------|
| `XML` | `.Format() = FileType::XML`   | enum | (autodetect extension `.xml`)  |
|-------------------------------------------------------------------------------|
| `BIN` | `.Format() = FileType::BIN`   | enum | (autodetect extension `.bin`)  |
|-------------------------------------------------------------------------------|

***Notes:***

 - `FileType::JSON` (`.json`) and `FileType::XML` (`.xml`) produce human-readable
   files, but they may be quite large.
 - `FileType::BIN` (`.bin`) is recommended for the sake of size; objects in
   binary FileType may be an order of magnitude or more smaller than JSON!

### [images](#images)

By default, loading and saving images is auto detected. If the user does not
want to specify the type, they can indicate that this an image by pasing
`Image` in the data option field. However, it is possible
to specify the file format we are trying to load / save. mlpack supports the
following formats only:

|-------------------------------------------------------------------------------|
| operator |  Function | Type  | Comment                                        |
|-------------------------------------------------------------------------------|
| `PNG` | `.Format() = FileType::PNG`   | enum | (autodetect extension `.png`)  |
|-------------------------------------------------------------------------------|
| `JPG` | `.Format() = FileType::JPG`   | enum | (autodetect extension `.jpg`)  |
|-------------------------------------------------------------------------------|
| `TGA` | `.format() = filetype::TGA`   | enum | (autodetect extension `.tga`)  |
|-------------------------------------------------------------------------------|
| `BMP` | `.format() = filetype::BMP`   | enum | (autodetect extension `.bmp`)  |
|-------------------------------------------------------------------------------|
| `gif` | `.format() = filetype::GIF`   | enum | (autodetect extension `.gif`)  |
|-------------------------------------------------------------------------------|
| `pic` | `.format() = filetype::PIC`   | enum | (autodetect extension `.pic`)  |
|-------------------------------------------------------------------------------|
| `pnm` | `.format() = filetype::PNM`   | enum | (autodetect extension `.pnm`)  |
|-------------------------------------------------------------------------------|
| `Image` | `.format() = filetype::ImageType` | enum | Not specifying the type  |
|-------------------------------------------------------------------------------|


## Matrix Options

During standard load / save operation we usually would like to Load the matrix
in column major (transposed). For this, MatrixOptions is going to be used when
we are Transpose operator as follows:

During standard load /save operations for plaintext formats (CSV/TSV/ASCII),
the following options allows the matrix to be transposed on load or save.
(Keep this `true` if you want | a column-major matrix to be loaded or saved
with points as rows and | dimensions as columns; that is generally what is desired.) 

|-------------------------------------------------------------------------------------------------------|
| Operator | Function | Type  | Comment | 
|-------------------------------------------------------------------------------------------------------|
| `Transpose`  | `.transpose() = true` | bool | The matrix will be transposed when load / save.         |
|-------------------------------------------------------------------------------------------------------|
| `NoTranspose`  | `.transpose() = false` | bool | The matrix will not be transposed when load / save.  |
|-------------------------------------------------------------------------------------------------------|

## Text Options

This class allows to specify settings related to the characteristic of the
matrix we are loading. For instance, does it contain categorical values? or
does the dataset has headers. The supported options are as following:

|-------------------------------------------------------------------------------------------------------|
| Operator | Function | Type  | Comment | 
|-------------------------------------------------------------------------------------------------------|
| `HasHeaders`   | `.HasHeaders()`  | bool | Set `true`, if the CSV file has a header, the header can   |
|                |                  |      | be accessible using `Headers()` function                   |
|--------------------------------------------------------------------------------------------------------
| `SemiColon`    | `.SemiColon()`   | bool | Set `true` for plaintext formats (CSV/TSV/ASCII) if the    |
|                |                  |      | separator is a semicolon instead of a comma                |
|--------------------------------------------------------------------------------------------------------
| `MissingToNan` | `.MissingToNan()`| bool | Set `true`, if there is missing data elements and you want |
|                |                  |      | them to be replaced with NaN                               |
|--------------------------------------------------------------------------------------------------------
| `Categorical`  | `.Categorical()` | bool | Set `true`, if the dataset contains categorical data.      |
|--------------------------------------------------------------------------------------------------------
