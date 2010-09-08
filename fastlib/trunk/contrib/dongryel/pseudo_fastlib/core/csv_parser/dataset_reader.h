/** @file dataset_reader.h
 *
 *  @author Dongryeol Lee
 */

#ifndef CORE_DATASET_READER_H
#define CORE_DATASET_READER_H

#include <string>
#include "core/csv_parser/csv_parser.h"

namespace core {
class DatasetReader {
  public:
    static void ParseDataset(
      const std::string &filename_in, arma::mat *dataset_out) {

      const char *filename = filename_in.c_str();
      const char field_terminator = ',';
      const char line_terminator  = '\n';
      const char enclosure_char   = '"';

      csv_parser file_parser;

      // Specify the number of lines to skip.
      file_parser.set_skip_lines(0);

      // Specify the file to parse.
      file_parser.init(filename);

      // Here we tell the parser how to parse the file.
      file_parser.set_enclosed_char(enclosure_char, ENCLOSURE_OPTIONAL);

      file_parser.set_field_term_char(field_terminator);

      file_parser.set_line_term_char(line_terminator);

      unsigned int num_points = 0;
      unsigned int num_dimensions = 0;

      // Check to see if there are more records, then grab each row
      // one at a time.
      while (file_parser.has_more_rows()) {
        unsigned int i = 0;

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
      while (file_parser.has_more_rows()) {
        unsigned int i = 0;

        // Get the record
        csv_row row = file_parser.get_row();
        num_dimensions = row.size();
        for (i = 0; i < row.size(); i++) {
          dataset_out->at(i, num_points) = atof(row[i].c_str());
        }
        num_points++;
      }
    }
};
};

#endif
