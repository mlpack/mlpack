/** @file dataset_reader.h
 *
 *  @author Dongryeol Lee
 */

#ifndef CORE_CSV_PARSER_DATASET_READER_H
#define CORE_CSV_PARSER_DATASET_READER_H

#include <string>
#include "core/csv_parser/csv_parser.h"
#include "core/math/math_lib.h"
#include "core/table/dense_matrix.h"
#include "core/table/memory_mapped_file.h"

namespace core {
class DatasetReader {

  private:

    template<typename TableType>
    static void Extract_(
      const std::string &filename_in,
      const std::string &filename_out,
      int num_dimensions_in, int begin, int end, int growupto) {

      int count = (growupto > 0) ? growupto : end - begin;
      TableType extracted_table;
      extracted_table.Init(num_dimensions_in, count);

      const char *filename = filename_in.c_str();
      const char field_terminator = ',';
      const char line_terminator  = '\n';
      const char enclosure_char   = '"';

      csv_parser file_parser;

      // Specify the number of lines to skip.
      file_parser.set_skip_lines(0);

      // Specify the file to parse.
      if(file_parser.init(filename) == false) {
        return;
      }

      // Here we tell the parser how to parse the file.
      file_parser.set_enclosed_char(enclosure_char, ENCLOSURE_OPTIONAL);
      file_parser.set_field_term_char(field_terminator);
      file_parser.set_line_term_char(line_terminator);

      int row_index = 0;
      int num_points = 0;
      while(file_parser.has_more_rows()) {

        // Get the record
        csv_row row = file_parser.get_row();

        if(row_index >= begin && row_index < end) {
          for(unsigned i = 0; i < row.size(); i++) {
            extracted_table.data().set(i, num_points, atof(row[i].c_str()));
          }
          num_points++;
        }
        row_index++;
      }
      for(; num_points < growupto; num_points++) {
        int random_point_index = core::math::RandInt(0, num_points);
        arma::vec source_point;
        arma::vec destination_point;
        extracted_table.get(random_point_index, &source_point);
        extracted_table.get(num_points, &destination_point);
        for(unsigned int j = 0; j < source_point.n_elem; j++) {
          destination_point[j] =
            source_point[j] +
            core::math::RandGaussian<double>(core::math::Random(0.1, 0.3));
        }
      }
      if(growupto > 0) {
        for(int i = 0; i < growupto; i++) {
          arma::vec point;
          extracted_table.get(i, &point);
          for(unsigned int j = 0; j < point.n_elem; j++) {
            point[j] +=
              core::math::RandGaussian<double>(core::math::Random(0.1, 0.3));
          }
        }
      }
      extracted_table.Save(filename_out);
    }

  public:

    template<typename TableType>
    static void GrowFile(
      const std::string &filename_in, int rank, int growupto) {

      // Count the number of points in total.
      int num_dimensions = 0;
      int total_num_points = CountNumPoints(filename_in, &num_dimensions);
      std::stringstream filename_out_sstr;
      filename_out_sstr << filename_in << rank;
      std::string filename_out = filename_out_sstr.str();
      Extract_<TableType>(
        filename_in, filename_out, num_dimensions,
        0, total_num_points, growupto);
    }

    template<typename TableType>
    static void SplitFile(
      const std::string &filename_in, int num_parts) {

      // Count the number of points in total.
      int num_dimensions = 0;
      int total_num_points = CountNumPoints(filename_in, &num_dimensions);
      int grain_size = total_num_points / num_parts;
      for(int i = 0; i < num_parts; i++) {
        int begin = i * grain_size;
        int end =
          (i < num_parts - 1) ? (i + 1) * grain_size : total_num_points;
        std::stringstream filename_out_sstr;
        filename_out_sstr << filename_in << i;
        std::string filename_out = filename_out_sstr.str();
        Extract_<TableType>(
          filename_in, filename_out, num_dimensions, begin, end, -1);
      }
    }

    /** @brief Counts the number of points in the dataset.
     */
    static int CountNumPoints(
      const std::string &filename_in, int *num_dimensions) {

      const char *filename = filename_in.c_str();
      const char field_terminator = ',';
      const char line_terminator  = '\n';
      const char enclosure_char   = '"';

      csv_parser file_parser;

      // Specify the number of lines to skip.
      file_parser.set_skip_lines(0);

      // Specify the file to parse.
      if(file_parser.init(filename) == false) {
        return false;
      }

      // Here we tell the parser how to parse the file.
      file_parser.set_enclosed_char(enclosure_char, ENCLOSURE_OPTIONAL);

      file_parser.set_field_term_char(field_terminator);

      file_parser.set_line_term_char(line_terminator);

      int num_points = 0;

      // Check to see if there are more records, then grab each row
      // one at a time.
      while(file_parser.has_more_rows()) {
        // Get the record
        csv_row row = file_parser.get_row();
        *num_dimensions = row.size();
        num_points++;
      }
      return num_points;
    }

    /** @brief Find the number of dimensions.
     */
    static int PeekNumDimensions(
      const std::string &filename_in) {

      const char *filename = filename_in.c_str();
      const char field_terminator = ',';
      const char line_terminator  = '\n';
      const char enclosure_char   = '"';

      csv_parser file_parser;

      // Specify the number of lines to skip.
      file_parser.set_skip_lines(0);

      // Specify the file to parse.
      if(file_parser.init(filename) == false) {
        return false;
      }

      // Here we tell the parser how to parse the file.
      file_parser.set_enclosed_char(enclosure_char, ENCLOSURE_OPTIONAL);

      file_parser.set_field_term_char(field_terminator);

      file_parser.set_line_term_char(line_terminator);

      // Get the record
      csv_row row = file_parser.get_row();
      return row.size();
    }

    static bool ParseDataset(
      const std::string &filename_in, core::table::DenseMatrix *dataset_out) {

      const char *filename = filename_in.c_str();
      const char field_terminator = ',';
      const char line_terminator  = '\n';
      const char enclosure_char   = '"';

      csv_parser file_parser;

      // Specify the number of lines to skip.
      file_parser.set_skip_lines(0);

      // Specify the file to parse.
      if(file_parser.init(filename) == false) {
        return false;
      }

      // Here we tell the parser how to parse the file.
      file_parser.set_enclosed_char(enclosure_char, ENCLOSURE_OPTIONAL);

      file_parser.set_field_term_char(field_terminator);

      file_parser.set_line_term_char(line_terminator);

      unsigned int num_points = 0;
      unsigned int num_dimensions = 0;

      // Check to see if there are more records, then grab each row
      // one at a time.
      while(file_parser.has_more_rows()) {
        // Get the record
        csv_row row = file_parser.get_row();
        num_dimensions = row.size();
        num_points++;
      }

      // Given the row count, allocate the matrix, while resetting the
      // file pointer.
      if(dataset_out->n_rows() == 0) {
        dataset_out->Init(num_dimensions, num_points);
      }
      file_parser.init(filename);

      // Grab each point and store.
      num_points = 0;
      while(file_parser.has_more_rows()) {
        // Get the record
        csv_row row = file_parser.get_row();
        num_dimensions = row.size();
        for(unsigned i = 0; i < row.size(); i++) {
          dataset_out->set(i, num_points, atof(row[i].c_str()));
        }
        num_points++;
      }

      // Assume that the dataset has been read.
      return true;
    }
};
};

#endif
