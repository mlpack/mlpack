/** @file dataset_reader.h
 *
 *  @author Dongryeol Lee
 */

#ifndef CORE_CSV_PARSER_DATASET_READER_H
#define CORE_CSV_PARSER_DATASET_READER_H

#include <armadillo>
#include <string>
#include "core/csv_parser/csv_parser.h"
#include "core/math/math_lib.h"
#include "core/table/memory_mapped_file.h"
#include "core/util/timer.h"

namespace core {
class DatasetReader {

  private:

    template<typename TableType>
    static void Extract_(
      const std::string &filename_out,
      int num_dimensions_in, int begin, int end,
      csv_parser *file_parser) {

      int count = end - begin;
      TableType extracted_table;
      extracted_table.Init(num_dimensions_in, count);

      // Extract the subset.
      for(int j = 0; j < count; j++) {

        // Get the record
        csv_row row = file_parser->get_row();
        for(unsigned i = 0; i < row.size(); i++) {
          extracted_table.data().at(i, j) = atof(row[i].c_str());
        }
      }
      extracted_table.Save(filename_out);
    }

  public:

    template<typename TableType>
    static void SplitFile(
      const std::string &filename_in, int num_parts) {

      core::util::Timer timer;
      timer.Start();

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
          filename_out, num_dimensions, begin, end, &file_parser);
      }

      timer.End();
      std::cout << timer.GetTotalElapsedTime() <<
                " seconds spent in splitting the file...\n";
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
      const std::string &filename_in, arma::mat *dataset_out) {

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
      dataset_out->set_size(num_dimensions, num_points);
      file_parser.init(filename);

      // Grab each point and store.
      num_points = 0;
      while(file_parser.has_more_rows()) {
        // Get the record
        csv_row row = file_parser.get_row();
        num_dimensions = row.size();
        for(unsigned i = 0; i < row.size(); i++) {
          dataset_out->at(i, num_points) = atof(row[i].c_str());
        }
        num_points++;
      }

      // Assume that the dataset has been read.
      return true;
    }
};
};

#endif
