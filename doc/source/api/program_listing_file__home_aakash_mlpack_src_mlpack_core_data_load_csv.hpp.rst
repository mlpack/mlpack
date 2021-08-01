
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_load_csv.hpp:

Program Listing for File load_csv.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_load_csv.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/load_csv.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_LOAD_CSV_HPP
   #define MLPACK_CORE_DATA_LOAD_CSV_HPP
   
   #include <boost/spirit/include/qi.hpp>
   #include <boost/algorithm/string/trim.hpp>
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/log.hpp>
   
   #include <set>
   #include <string>
   
   #include "extension.hpp"
   #include "format.hpp"
   #include "dataset_mapper.hpp"
   
   namespace mlpack {
   namespace data {
   
   class LoadCSV
   {
    public:
     LoadCSV(const std::string& file);
   
     template<typename T, typename PolicyType>
     void Load(arma::Mat<T> &inout,
               DatasetMapper<PolicyType> &infoSet,
               const bool transpose = true)
     {
       CheckOpen();
   
       if (transpose)
         TransposeParse(inout, infoSet);
       else
         NonTransposeParse(inout, infoSet);
     }
   
     template<typename T, typename MapPolicy>
     void GetMatrixSize(size_t& rows, size_t& cols, DatasetMapper<MapPolicy>& info)
     {
       using namespace boost::spirit;
   
       // Take a pass through the file.  If the DatasetMapper policy requires it,
       // we will pass everything string through MapString().  This might be useful
       // if, e.g., the MapPolicy needs to find which dimensions are numeric or
       // categorical.
   
       // Reset to the start of the file.
       inFile.clear();
       inFile.seekg(0, std::ios::beg);
       rows = 0;
       cols = 0;
   
       // First, count the number of rows in the file (this is the dimensionality).
       std::string line;
       while (std::getline(inFile, line))
       {
         ++rows;
       }
       info = DatasetMapper<MapPolicy>(rows);
   
       // Now, jump back to the beginning of the file.
       inFile.clear();
       inFile.seekg(0, std::ios::beg);
       rows = 0;
   
       while (std::getline(inFile, line))
       {
         ++rows;
         // Remove whitespace from either side.
         boost::trim(line);
   
         if (rows == 1)
         {
           // Extract the number of columns.
           auto findColSize = [&cols](iter_type) { ++cols; };
           qi::parse(line.begin(), line.end(),
               stringRule[findColSize] % delimiterRule);
         }
   
         // I guess this is technically a second pass, but that's ok... still the
         // same idea...
         if (MapPolicy::NeedsFirstPass)
         {
           // In this case we must pass everything we parse to the MapPolicy.
           auto firstPassMap = [&](const iter_type& iter)
           {
             std::string str(iter.begin(), iter.end());
             boost::trim(str);
   
             info.template MapFirstPass<T>(std::move(str), rows - 1);
           };
   
           // Now parse the line.
           qi::parse(line.begin(), line.end(),
               stringRule[firstPassMap] % delimiterRule);
         }
       }
     }
   
     template<typename T, typename MapPolicy>
     void GetTransposeMatrixSize(size_t& rows,
                                 size_t& cols,
                                 DatasetMapper<MapPolicy>& info)
     {
       using namespace boost::spirit;
   
       // Take a pass through the file.  If the DatasetMapper policy requires it,
       // we will pass everything string through MapString().  This might be useful
       // if, e.g., the MapPolicy needs to find which dimensions are numeric or
       // categorical.
   
       // Reset to the start of the file.
       inFile.clear();
       inFile.seekg(0, std::ios::beg);
       rows = 0;
       cols = 0;
   
       std::string line;
       while (std::getline(inFile, line))
       {
         ++cols;
         // Remove whitespace from either side.
         boost::trim(line);
   
         if (cols == 1)
         {
           // Extract the number of dimensions.
           auto findRowSize = [&rows](iter_type) { ++rows; };
           qi::parse(line.begin(), line.end(),
               stringRule[findRowSize] % delimiterRule);
   
           // Now that we know the dimensionality, initialize the DatasetMapper.
           info.SetDimensionality(rows);
         }
   
         // If we need to do a first pass for the DatasetMapper, do it.
         if (MapPolicy::NeedsFirstPass)
         {
           size_t dim = 0;
   
           // In this case we must pass everything we parse to the MapPolicy.
           auto firstPassMap = [&](const iter_type& iter)
           {
             std::string str(iter.begin(), iter.end());
             boost::trim(str);
   
             info.template MapFirstPass<T>(std::move(str), dim++);
           };
   
           // Now parse the line.
           qi::parse(line.begin(), line.end(),
               stringRule[firstPassMap] % delimiterRule);
         }
       }
     }
   
    private:
     using iter_type = boost::iterator_range<std::string::iterator>;
   
     void CheckOpen();
   
     template<typename T, typename PolicyType>
     void NonTransposeParse(arma::Mat<T>& inout,
                            DatasetMapper<PolicyType>& infoSet)
     {
       using namespace boost::spirit;
   
       // Get the size of the matrix.
       size_t rows, cols;
       GetMatrixSize<T>(rows, cols, infoSet);
   
       // Set up output matrix.
       inout.set_size(rows, cols);
       size_t row = 0;
       size_t col = 0;
   
       // Reset file position.
       std::string line;
       inFile.clear();
       inFile.seekg(0, std::ios::beg);
   
       auto setCharClass = [&](iter_type const &iter)
       {
         std::string str(iter.begin(), iter.end());
         if (str == "\t")
         {
           str.clear();
         }
         boost::trim(str);
   
         inout(row, col++) = infoSet.template MapString<T>(std::move(str), row);
       };
   
       while (std::getline(inFile, line))
       {
         // Remove whitespace from either side.
         boost::trim(line);
   
         // Parse the numbers from a line (ex: 1,2,3,4); if the parser finds a
         // number it will execute the setNum function.
         const bool canParse = qi::parse(line.begin(), line.end(),
             stringRule[setCharClass] % delimiterRule);
   
         // Make sure we got the right number of rows.
         if (col != cols)
         {
           std::ostringstream oss;
           oss << "LoadCSV::NonTransposeParse(): wrong number of dimensions ("
               << col << ") on line " << row << "; should be " << cols
               << " dimensions.";
           throw std::runtime_error(oss.str());
         }
   
         if (!canParse)
         {
           std::ostringstream oss;
           oss << "LoadCSV::NonTransposeParse(): parsing error on line " << col
               << "!";
           throw std::runtime_error(oss.str());
         }
   
         ++row; col = 0;
       }
     }
   
     template<typename T, typename PolicyType>
     void TransposeParse(arma::Mat<T>& inout, DatasetMapper<PolicyType>& infoSet)
     {
       using namespace boost::spirit;
   
       // Get matrix size.  This also initializes infoSet correctly.
       size_t rows, cols;
       GetTransposeMatrixSize<T>(rows, cols, infoSet);
   
       // Set the matrix size.
       inout.set_size(rows, cols);
   
       // Initialize auxiliary variables.
       size_t row = 0;
       size_t col = 0;
       std::string line;
       inFile.clear();
       inFile.seekg(0, std::ios::beg);
   
       auto parseString = [&](iter_type const &iter)
       {
         // All parsed values must be mapped.
         std::string str(iter.begin(), iter.end());
         boost::trim(str);
   
         inout(row, col) = infoSet.template MapString<T>(std::move(str), row);
         ++row;
       };
   
       while (std::getline(inFile, line))
       {
         // Remove whitespace from either side.
         boost::trim(line);
   
         // Reset the row we are looking at.  (Remember this is transposed.)
         row = 0;
   
         // Now use boost::spirit to parse the characters of the line;
         // parseString() will be called when a token is detected.
         const bool canParse = qi::parse(line.begin(), line.end(),
             stringRule[parseString] % delimiterRule);
   
         // Make sure we got the right number of rows.
         if (row != rows)
         {
           std::ostringstream oss;
           oss << "LoadCSV::TransposeParse(): wrong number of dimensions (" << row
               << ") on line " << col << "; should be " << rows << " dimensions.";
           throw std::runtime_error(oss.str());
         }
   
         if (!canParse)
         {
           std::ostringstream oss;
           oss << "LoadCSV::TransposeParse(): parsing error on line " << col
               << "!";
           throw std::runtime_error(oss.str());
         }
   
         // Increment the column index.
         ++col;
       }
     }
   
     boost::spirit::qi::rule<std::string::iterator, iter_type()> stringRule;
     boost::spirit::qi::rule<std::string::iterator, iter_type()> delimiterRule;
   
     std::string extension;
     std::string filename;
     std::ifstream inFile;
   };
   
   } // namespace data
   } // namespace mlpack
   
   #endif
