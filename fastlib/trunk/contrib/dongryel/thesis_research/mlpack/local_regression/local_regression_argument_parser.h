/** @file local_regression_argument_parser.h
 *
 *  The argument parsing for local regression is handled here.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_ARGUMENT_PARSER_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_ARGUMENT_PARSER_H

#include <boost/program_options.hpp>
#include "core/table/transform.h"
#include "mlpack/local_regression/local_regression_arguments.h"

namespace mlpack {
namespace local_regression {

/** @brief The class for parsing the necessary arguments for the local
 *         regression.
 */
class LocalRegressionArgumentParser {
  public:
    static bool ConstructBoostVariableMap(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm) {

      boost::program_options::options_description desc("Available options");
      desc.add_options()(
        "help", "Print this information."
      )(
        "absolute_error",
        boost::program_options::value<double>()->default_value(0.0),
        "The absolute error level in approximation."
      )(
        "bandwidth",
        boost::program_options::value<double>(),
        "REQUIRED kernel bandwidth"
      )(
        "kernel",
        boost::program_options::value<std::string>()->default_value("gaussian"),
        "Kernel function used by local regression.  One of:\n"
        "  epan, gaussian"
      )(
        "leaf_size",
        boost::program_options::value<int>()->default_value(20),
        "Maximum number of points at a leaf of the tree."
      )(
        "metric",
        boost::program_options::value<std::string>()->default_value("lmetric"),
        "Metric type used by local regression.  One of:\n"
        "  lmetric, weighted_lmetric"
      )(
        "metric_scales_in",
        boost::program_options::value<std::string>(),
        "The file containing the scaling factors for the weighted L2 metric."
      )(
        "order",
        boost::program_options::value<int>()->default_value(0),
        "The order of local polynomial to fit at each query point. One of: "
        "  0 or 1"
      )(
        "predictions_out",
        boost::program_options::value<std::string>()->default_value(
          "predictions_out.csv"),
        "OPTIONAL file to store the predicted regression values."
      )(
        "prescale",
        boost::program_options::value<std::string>()->default_value("none"),
        "OPTIONAL scaling option. One of:\n"
        "  none, hypercube, standardize"
      )(
        "probability",
        boost::program_options::value<double>()->default_value(1.0),
        "The required probability guarantee level.\n"
      )(
        "queries_in",
        boost::program_options::value<std::string>(),
        "OPTIONAL file containing query positions.  If omitted, local "
        "regression computes the leave-one-out density at each reference point."
      )(
        "references_in",
        boost::program_options::value<std::string>(),
        "The file containing the reference points."
      )(
        "relative_error",
        boost::program_options::value<double>()->default_value(0.1),
        "Relative error for the approximation of local regression."
      )(
        "reference_targets_in",
        boost::program_options::value<std::string>(),
        "The file containing the target values for the reference set."
      );

      boost::program_options::command_line_parser clp(args);
      clp.style(boost::program_options::command_line_style::default_style
                ^ boost::program_options::command_line_style::allow_guessing);
      try {
        boost::program_options::store(clp.options(desc).run(), *vm);
      }
      catch(const boost::program_options::invalid_option_value &e) {
        std::cerr << "Invalid Argument: " << e.what() << "\n";
        exit(0);
      }
      catch(const boost::program_options::invalid_command_line_syntax &e) {
        std::cerr << "Invalid command line syntax: " << e.what() << "\n";
        exit(0);
      }
      catch(const boost::program_options::unknown_option &e) {
        std::cerr << "Unknown option: " << e.what() << "\n";
        exit(0);
      }

      boost::program_options::notify(*vm);
      if(vm->count("help")) {
        std::cerr << desc << "\n";
        return true;
      }

      // Validate the arguments. Only immediate quitting is allowed
      // here, the parsing is done later.
      if(vm->count("bandwidth") > 0 && (*vm)["bandwidth"].as<double>() <= 0) {
        std::cerr << "The --bandwidth requires a positive real number.\n";
        exit(0);
      }
      if(vm->count("bandwidth") == 0) {
        std::cerr << "Missing required --bandwidth.\n";
        exit(0);
      }
      if((*vm)["kernel"].as<std::string>() != "gaussian" &&
          (*vm)["kernel"].as<std::string>() != "epan") {
        std::cerr << "We support only epan or gaussian for the kernel.\n";
        exit(0);
      }
      if((*vm)["leaf_size"].as<int>() <= 0) {
        std::cerr << "The --leaf_size requires a positive integer.\n";
        exit(0);
      }
      if((*vm)["metric"].as<std::string>() == "weighted_lmetric" &&
          vm->count("metric_scales_in") == 0) {
        std::cerr << "The weighted L2 metric option requires " <<
                  "--metric_scales_in option.\n";
        exit(0);
      }
      if((*vm)["order"].as<int>() < 0 || (*vm)["order"].as<int>() > 1) {
        std::cerr << "The --order requires either 0 (Nadaraya-Watson) or "
                  "1 (local-linear).\n";
        exit(0);
      }
      if((*vm)["prescale"].as<std::string>() != "none" &&
          (*vm)["prescale"].as<std::string>() != "hypercube" &&
          (*vm)["prescale"].as<std::string>() != "standardize") {
        std::cerr << "The --prescale supports only one of: none, hypercube, " <<
                  "standardize.\n";
        exit(0);
      }
      if((*vm)["probability"].as<double>() <= 0.0) {
        std::cerr << "The --probability requires a non-negative real number.\n";
        exit(0);
      }

      if(vm->count("references_in") == 0) {
        std::cerr << "Missing required --references_in.\n";
        exit(0);
      }
      if(vm->count("reference_targets_in") == 0) {
        std::cerr << "Missing reqiured --reference_targets_in.\n";
        exit(0);
      }
      if((*vm)["absolute_error"].as<double>() < 0.0) {
        std::cerr << "The --absolute_error requires a non-negative real "
                  << "number.\n";
        exit(0);
      }
      if((*vm)["relative_error"].as<double>() < 0.0) {
        std::cerr << "The --relative_error requires a non-negative real "
                  << "number.\n";
        exit(0);
      }

      return false;
    }

    static bool ConstructBoostVariableMap(
      int argc, char *argv[],
      boost::program_options::variables_map *vm) {

      // Convert C input to C++; skip executable name for Boost.
      std::vector<std::string> args(argv + 1, argv + argc);

      return ConstructBoostVariableMap(args, vm);
    }

    template<typename TableType, typename MetricType>
    static bool ParseArguments(
      const std::vector<std::string> &args,
      mlpack::local_regression::LocalRegressionArguments <
      TableType, MetricType > *arguments_out) {

      // Construct the Boost variable map.
      boost::program_options::variables_map vm;
      if(ConstructBoostVariableMap(args, &vm)) {
        exit(0);
      }

      return ParseArguments(vm, arguments_out);
    }

    template<typename TableType>
    static void PrescaleTable_(
      const std::string &prescale_option, TableType *table) {

      if(prescale_option == "hypercube") {
        core::table::UnitHypercube::Transform(table);
      }
      else if(prescale_option == "standardize") {
        core::table::Standardize::Transform(table);
      }

      // Now, make sure that all coordinates are non-negative.
      if(prescale_option != "hypercube") {
        core::table::TranslateToNonnegative::Transform(table);
      }
    }

    template<typename TableType, typename MetricType>
    static bool ParseArguments(
      boost::program_options::variables_map &vm,
      mlpack::local_regression::LocalRegressionArguments <
      TableType, MetricType > *arguments_out) {

      // Parse bandwidth.
      arguments_out->bandwidth_ =  vm["bandwidth"].as<double>();
      std::cerr << "Bandwidth: " << arguments_out->bandwidth_ << "\n";

      // Parse the kernel.
      arguments_out->kernel_ = vm["kernel"].as<std::string>();
      std::cerr << "Kernel: " << arguments_out->kernel_ << "\n";

      // Parse the leaf size.
      arguments_out->leaf_size_ = vm["leaf_size"].as<int>();
      std::cerr << "Leaf size: " << arguments_out->leaf_size_ << "\n";

      // Parse the metric weights file.
      std::cerr << "Metric type: " << vm["metric"].as<std::string>() << "\n";
      if(vm["metric"].as<std::string>() == "weighted_lmetric") {
        TableType metric_scales_in;
        std::cerr << "Reading in the metric scales from: " <<
                  vm["metric_scales_in"].as<std::string>() << "\n";
        metric_scales_in.Init(vm["metric_scales_in"].as<std::string>());
        arguments_out->metric_.set_scales(metric_scales_in);
      }

      // Parse the predictions out file.
      arguments_out->predictions_out_ = vm["predictions_out"].as<std::string>();

      // Parse the prescale option.
      arguments_out->prescale_ = vm["prescale"].as<std::string>();
      std::cerr << "The prescale option: " << arguments_out->prescale_ << "\n";

      // Read in the reference set.
      arguments_out->reference_table_ = new TableType();
      arguments_out->reference_table_->Init(
        vm["references_in"].as<std::string>(), 0,
        &(vm["reference_targets_in"].as<std::string>()));
      std::cerr << "Read in " << arguments_out->reference_table_->n_entries() <<
                " points in " <<
                arguments_out->reference_table_->n_attributes() <<
                " dimensions.\n";

      // Parse the local polynomial order.
      arguments_out->order_ = vm["order"].as<int>();
      arguments_out->problem_dimension_ =
        (arguments_out->order_ == 0) ?
        1 : arguments_out->reference_table_->n_attributes() + 1;
      std::cerr << "The requested local polynomial order: " <<
                arguments_out->order_ << "\n";

      // Scale the dataset.
      PrescaleTable_(
        vm["prescale"].as<std::string>(), arguments_out->reference_table_);
      std::cerr << "Scaled the dataset with the option: " <<
                vm["prescale"].as<std::string>() << "\n";

      std::cerr << "Building the reference tree.\n";
      arguments_out->reference_table_->IndexData(
        arguments_out->metric_, arguments_out->leaf_size_);
      std::cerr << "Finished building the reference tree.\n";

      // Read in the probability.
      arguments_out->probability_ = vm["probability"].as<double>();
      std::cerr << "The probability guarantee requested: " <<
                arguments_out->probability_ << "\n";

      // Read in the optional query set.
      if(vm.count("queries_in") > 0) {
        arguments_out->query_table_ = new TableType();
        arguments_out->query_table_->Init(vm["queries_in"].as<std::string>());
        std::cerr << "Read in " << arguments_out->query_table_->n_entries() <<
                  " points in " <<
                  arguments_out->query_table_->n_attributes()
                  << " dimensions.\n";

        // Scale the dataset.
        PrescaleTable_(
          vm["prescale"].as<std::string>(), arguments_out->query_table_);
        std::cerr << "Scaled the dataset with the option: " <<
                  vm["prescale"].as<std::string>() << "\n";

        std::cerr << "Building the query tree.\n";
        arguments_out->query_table_->IndexData(
          arguments_out->metric_, arguments_out->leaf_size_);
        std::cerr << "Finished building the query tree.\n";
        arguments_out->effective_num_reference_points_ =
          arguments_out->reference_table_->n_entries();
        arguments_out->is_monochromatic_ = false;
      }
      else {
        arguments_out->query_table_ = arguments_out->reference_table_;
        arguments_out->effective_num_reference_points_ =
          arguments_out->reference_table_->n_entries() - 1;
        arguments_out->is_monochromatic_ = true;
      }

      // Read in the absolute error.
      arguments_out->absolute_error_ = vm["absolute_error"].as<double>();
      std::cerr << "The absolute error requested: " <<
                arguments_out->absolute_error_ << "\n";

      // Read in the relative error.
      arguments_out->relative_error_ = vm["relative_error"].as<double>();
      std::cerr << "The relative error requested: " <<
                arguments_out->relative_error_ << "\n";

      return false;
    }

    template<typename TableType, typename MetricType>
    static bool ParseArguments(
      int argc,
      char *argv[],
      mlpack::local_regression::LocalRegressionArguments <
      TableType, MetricType > *arguments_out) {

      // Convert C input to C++; skip executable name for Boost.
      std::vector<std::string> args(argv + 1, argv + argc);

      return ParseArguments(args, arguments_out);
    }
};
}
}

#endif
