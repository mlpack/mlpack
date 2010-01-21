#ifndef FL_LITE_MLPACK_REGRESSION_GIVENS_ROTATE_H
#define FL_LITE_MLPACK_REGRESSION_GIVENS_ROTATE_H

class GivensRotate {

  public:

    /** @brief Applies the Givens rotation row-wise.
     */
    template<typename MatrixType>
    static void ApplyToRow(double cosine_value, double sine_value,
                           int first_row_index, int second_row_index,
                           MatrixType &matrix);

    /** @brief Applies the Givens rotation column-wise.
     */
    template<typename MatrixType>
    static void ApplyToColumn(double cosine_value, double sine_value,
                              int first_column_index, int second_column_index,
                              MatrixType &matrix);

    /** @brief Computes the Givens rotation such that the second
     *         value becomes zero.
     */
    static void Compute(double first, double second,
                        double *magnitude, double *cosine_value,
                        double *sine_value);
};

#endif
