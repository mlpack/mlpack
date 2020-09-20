# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  cross_entropy_error.hpp
  cross_entropy_error_impl.hpp
  cosine_embedding_loss.hpp
  cosine_embedding_loss_impl.hpp
  dice_loss.hpp
  dice_loss_impl.hpp
  earth_mover_distance.hpp
  earth_mover_distance_impl.hpp
  huber_loss.hpp
  huber_loss_impl.hpp
  kl_divergence.hpp
  kl_divergence_impl.hpp
  margin_ranking_loss.hpp
  margin_ranking_loss_impl.hpp
  mean_bias_error.hpp
  mean_bias_error_impl.hpp
  l1_loss.hpp
  l1_loss_impl.hpp
  mean_squared_error.hpp
  mean_squared_error_impl.hpp
  mean_squared_logarithmic_error.hpp
  mean_squared_logarithmic_error_impl.hpp
  negative_log_likelihood.hpp
  negative_log_likelihood_impl.hpp
  log_cosh_loss.hpp
  log_cosh_loss_impl.hpp
  poisson_nll_loss.hpp
  poisson_nll_loss_impl.hpp
  reconstruction_loss.hpp
  reconstruction_loss_impl.hpp
  sigmoid_cross_entropy_error.hpp
  sigmoid_cross_entropy_error_impl.hpp
  soft_margin_loss.hpp
  soft_margin_loss_impl.hpp
  hinge_embedding_loss.hpp
  hinge_embedding_loss_impl.hpp
  empty_loss.hpp
  empty_loss_impl.hpp
  mean_absolute_percentage_error.hpp
  mean_absolute_percentage_error_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
