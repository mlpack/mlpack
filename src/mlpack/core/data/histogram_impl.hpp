/**
 * @file core/data/histogram.hpp
 * @author Aditi Pandey
 *
 * Histogram function. The purpose of this function is to plot graph w.r.t 
 * given x and y list of inputs.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_HISTOGRAM_IMPL_HPP
#define MLPACK_CORE_DATA_HISTOGRAM_IMPL_HPP

// In case it hasn't been included yet.
#include "histogram.hpp"

namespace mlpack {
namespace data {


/**
 * Function to plot histogram 
 * User should provide at least the required parameters
 *
 * REQUIRED:
 * @param independent_var is the Input list which will provide the independent variable usually known as x.
 * @param dependent_var is the Input list which will provide the dependent variable usually known as y.

 *
 * EXTRA PARAMETERS: 
 * @param plot_lim_left the lowest number limit till the x-axis should be plotted on the graph.
 * @param plot_lim_right the highest number limit till the x-axis should be plotted on the graph.
 * @param size_len number of pixels in x-dimentions to be plotted in the output.
 * @param size_breath number of pixels in y-dimentions to be plotted in the output.
 * @param legend_name the name that should be given to the line in the plot.
 * @param plot_title the title that should be provided to the plot.
 * @param plot_name the name with the extension that the plot should be saved with.
 */
template<typename eT>
void Histogram(const arma::Row<cx_double>& independent_var,
                    const arma::Col<cx_double>& dependent_var,
                    int plot_lim_left = 0, int plot_lim_right = 100,
                    int size_len = 1200, int size_breath = 780,
                    string legend_name = "Legend Name",
                    string plot_title = "Plot Title";
                    string plot_name = "./Plot_name.png";
                    )
 {


    // Set the size of output image to 1200x780 pixels by default
    plt::figure_size(size_len, size_breath);
    // Plot line from given independent variable and dependent variable data. 
    // Color is selected automatically.
    // Plot line will show up as "Legend Name" in the legend by default.
    plt::named_plot(legend_name, independent_var, dependent_var);
    // Set x-axis to interval [0,100] by default.
    plt::xlim(plot_lim_left, plot_lim_right);
    // Add graph title
    plt::title(plot_title);
    // Enable legend.
    plt::legend();
    // Save the image (file format is determined by the extension)
    // Using "./Plot_name.png" by default
    plt::save(plot_name);
 }

} // namespace data
} // namespace mlpack

#endif
