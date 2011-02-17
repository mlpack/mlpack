/** @file mixed_logit_dcm_argument_parser.h
 *
 *  The argument parsing for mixed logit discrete choice model is
 *  handled here.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_ARGUMENT_PARSER_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_ARGUMENT_PARSER_H

#include <boost/program_options.hpp>
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_arguments.h"

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The class for parsing the necessary arguments for the
 *         mixed logit discrete choice model.
 */
class MixedLogitDCMArgumentParser {
  private:
    static bool ConstructBoostVariableMap_(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm) {

      boost::program_options::options_description desc("Available options");
      desc.add_options()(
        "help", "Print this information."
      )(
        "attributes_in",
        boost::program_options::value<std::string>(),
        "REQUIRED file containing the vector of attributes."
      )(
        "gradient_norm_threshold",
        boost::program_options::value<double>()->default_value(0.5),
        "OPTIONAL The threshold for determining the termination condition "
        "based on the gradient norm."
      )(
        "hessian_update_method",
        boost::program_options::value<std::string>()->default_value("exact"),
        "OPTIONAL The method for updating the Hessian. One of:"
        "  exact, bfgs, sr1"
      )(
        "initial_dataset_sample_rate",
        boost::program_options::value<double>()->default_value(0.1),
        "OPTIONAL the rate at which to sample the entire dataset in the "
        "beginning."
      )(
        "initial_integration_sample_rate",
        boost::program_options::value<double>()->default_value(0.01),
        "OPTIONAL The percentage of the maximum average integration sample "
        "to start with."
      )(
        "integration_sample_error_threshold",
        boost::program_options::value<double>()->default_value(1e-9),
        "OPTIONAL The threshold for determining whether the integration sample "
        "error is small or not."
      )(
        "max_num_integration_samples_per_person",
        boost::program_options::value<int>()->default_value(1000),
        "OPTIONAL The maximum number of integration samples allowed per person."
      )(
        "max_trust_region_radius",
        boost::program_options::value<double>()->default_value(10.0),
        "OPTIONAL The maximum trust region radius used in the trust region "
        "search."
      )(
        "num_discrete_choices_per_person_in",
        boost::program_options::value<std::string>(),
        "REQUIRED The number of alternatives per each person."
      )(
        "predictions_out",
        boost::program_options::value<std::string>()->default_value(
          "densities_out.csv"),
        "OPTIONAL file to store the predicted discrete choices."
      )(
        "trust_region_search_method",
        boost::program_options::value<std::string>()->default_value("cauchy"),
        "OPTIONAL Trust region search method.  One of:\n"
        "  cauchy, dogleg, steihaug"
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
        std::cout << desc << "\n";
        return true;
      }

      // Validate the arguments. Only immediate quitting is allowed here,
      // the parsing is done later.
      if(vm->count("attributes_in") == 0) {
        std::cerr << "Missing required --attributes_in.\n";
        exit(0);
      }
      if(vm->count("num_discrete_choices_per_person_in") == 0) {
        std::cerr << "Missing required --num_discrete_choices_per_person_in.\n";
        exit(0);
      }
      if((*vm)["trust_region_search_method"].as<std::string>() != "cauchy" &&
          (*vm)["trust_region_search_method"].as<std::string>() != "dogleg" &&
          (*vm)["trust_region_search_method"].as<std::string>() != "steihaug") {
        std::cerr << "Invalid option specified for " <<
                  "--trust_region_search_method.\n";
        exit(0);
      }
      if((*vm)["hessian_update_method"].as<std::string>() != "none" &&
          (*vm)["hessian_update_method"].as<std::string>() != "bfgs" &&
          (*vm)["hessian_update_method"].as<std::string>() != "sr1") {
        std::cerr << "Invalid option specified for " <<
                  "--hessian_update_method.\n";
        exit(0);
      }

      return false;
    }

  public:

    template<typename TableType>
    static void ParseArguments(
      const std::vector<std::string> &args,
      mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
      TableType > *arguments_out) {

      // Construct the Boost variable map.
      boost::program_options::variables_map vm;
      if(ConstructBoostVariableMap_(args, &vm)) {
        exit(0);
      }

      // Given the constructed boost variable map, parse each argument.

      // Parse the set of attribute vectors.
      std::cout << "Reading in the attribute set: " <<
                vm["attributes_in"].as<std::string>() << "\n";
      arguments_out->attribute_table_ = new TableType();
      arguments_out->attribute_table_->Init(
        vm["attributes_in"].as<std::string>());
      std::cout << "Finished reading in the attributes set.\n";

      // Parse the number of discrete choices per each point.
      std::cout <<
                "Reading in the number of discrete choices per each person: " <<
                vm["num_discrete_choices_per_person_in"].as<std::string>() <<
                "\n";
      arguments_out->num_discrete_choices_per_person_ = new TableType();
      arguments_out->num_discrete_choices_per_person_->Init(
        vm["num_discrete_choices_per_person_in"].as<std::string>());

      // Parse the initial dataset sample rate.
      arguments_out->initial_dataset_sample_rate_ =
        vm[ "initial_dataset_sample_rate" ].as<double>();

      // Parse the initial integration sample rate.
      arguments_out->initial_integration_sample_rate_ =
        vm[ "initial_integration_sample_rate" ].as<double>();

      // Parse the gradient norm threshold.
      arguments_out->gradient_norm_threshold_ =
        vm[ "gradient_norm_threshold" ].as<double>();

      // Parse the maximum number of integration samples per person.
      arguments_out->max_num_integration_samples_per_person_ =
        vm[ "max_num_integration_samples_per_person" ].as<int>();

      // Parse the integration sample error thresold.
      arguments_out->integration_sample_error_threshold_ =
        vm["integration_sample_error_threshold"].as<double>();

      // Parse where to output the discrete choice model predictions.
      arguments_out->predictions_out_ =
        vm[ "predictions_out" ].as<std::string>();

      // Parse the maximum trust region radius.
      arguments_out->max_trust_region_radius_ =
        vm[ "max_trust_region_radius" ].as<double>();

      // Parse how to perform the trust region search: cauchy, dogleg, steihaug
      if(vm[ "trust_region_search_method" ].as<std::string>() == "cauchy") {
        arguments_out->trust_region_search_method_ =
          core::optimization::TrustRegionSearchMethod::CAUCHY;
      }
      else if(vm["trust_region_search_method"].as<std::string>() == "dogleg") {
        arguments_out->trust_region_search_method_ =
          core::optimization::TrustRegionSearchMethod::DOGLEG;
      }
      else {
        arguments_out->trust_region_search_method_ =
          core::optimization::TrustRegionSearchMethod::STEIHAUG;
      }

      // Parse the Hessian update method.
      arguments_out->hessian_update_method_ =
        vm["hessian_update_method"].as<std::string>();

      // The number of parameters that generate each $\beta$ is fixed now
      // as the Gaussian example in Appendix.
      // arguments_out->distribution_ = ;
    }

    template<typename TableType>
    static void ParseArguments(
      int argc,
      char *argv[],
      mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
      TableType > *arguments_out) {

      // Convert C input to C++; skip executable name for Boost.
      std::vector<std::string> args(argv + 1, argv + argc);

      ParseArguments(args, arguments_out);
    }
};
}
}

#endif
