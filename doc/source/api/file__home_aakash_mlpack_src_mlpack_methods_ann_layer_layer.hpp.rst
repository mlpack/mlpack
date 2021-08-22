
.. _file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer.hpp:

File layer.hpp
==============

|exhale_lsh| :ref:`Parent directory <dir__home_aakash_mlpack_src_mlpack_methods_ann_layer>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. contents:: Contents
   :local:
   :backlinks: none

Definition (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/layer.hpp``)
---------------------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer.hpp.rst



Detailed Description
--------------------

Marcus Edel
This includes various layers to construct a model.
mlpack is free software; you may redistribute it and/or modify it under the terms of the 3-clause BSD license. You should have received a copy of the 3-clause BSD license along with mlpack. If not, see http://www.opensource.org/licenses/BSD-3-Clause for more information. 



Includes
--------


- ``adaptive_max_pooling.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_adaptive_max_pooling.hpp`)

- ``adaptive_mean_pooling.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_adaptive_mean_pooling.hpp`)

- ``add.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_augmented_tasks_add.hpp`)

- ``add_merge.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add_merge.hpp`)

- ``alpha_dropout.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_alpha_dropout.hpp`)

- ``atrous_convolution.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_atrous_convolution.hpp`)

- ``base_layer.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_base_layer.hpp`)

- ``batch_norm.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_batch_norm.hpp`)

- ``bicubic_interpolation.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_bicubic_interpolation.hpp`)

- ``bilinear_interpolation.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_bilinear_interpolation.hpp`)

- ``c_relu.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_c_relu.hpp`)

- ``celu.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_celu.hpp`)

- ``concat.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concat.hpp`)

- ``concat_performance.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concat_performance.hpp`)

- ``concatenate.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concatenate.hpp`)

- ``constant.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_constant.hpp`)

- ``convolution.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_atrous_convolution.hpp`)

- ``dropconnect.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_dropconnect.hpp`)

- ``dropout.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_alpha_dropout.hpp`)

- ``elu.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_c_relu.hpp`)

- ``fast_lstm.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_fast_lstm.hpp`)

- ``flatten_t_swish.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_flatten_t_swish.hpp`)

- ``flexible_relu.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_flexible_relu.hpp`)

- ``glimpse.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_glimpse.hpp`)

- ``gru.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_gru.hpp`)

- ``hard_tanh.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hard_tanh.hpp`)

- ``hardshrink.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hardshrink.hpp`)

- ``highway.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_highway.hpp`)

- ``join.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_join.hpp`)

- ``layer_norm.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer_norm.hpp`)

- ``layer_types.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer_types.hpp`)

- ``leaky_relu.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_leaky_relu.hpp`)

- ``linear.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_linear.hpp`)

- ``linear3d.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_linear3d.hpp`)

- ``linear_no_bias.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_linear_no_bias.hpp`)

- ``log_softmax.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_log_softmax.hpp`)

- ``lookup.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lookup.hpp`)

- ``lp_pooling.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lp_pooling.hpp`)

- ``lstm.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_fast_lstm.hpp`)

- ``max_pooling.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_adaptive_max_pooling.hpp`)

- ``mean_pooling.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_adaptive_mean_pooling.hpp`)

- ``minibatch_discrimination.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_minibatch_discrimination.hpp`)

- ``multihead_attention.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multihead_attention.hpp`)

- ``multiply_constant.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_constant.hpp`)

- ``multiply_merge.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_merge.hpp`)

- ``nearest_interpolation.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_nearest_interpolation.hpp`)

- ``noisylinear.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_noisylinear.hpp`)

- ``padding.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_padding.hpp`)

- ``parametric_relu.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_parametric_relu.hpp`)

- ``pixel_shuffle.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_pixel_shuffle.hpp`)

- ``positional_encoding.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_positional_encoding.hpp`)

- ``recurrent.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_recurrent.hpp`)

- ``recurrent_attention.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_recurrent_attention.hpp`)

- ``reinforce_normal.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_reinforce_normal.hpp`)

- ``relu6.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_relu6.hpp`)

- ``reparametrization.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_reparametrization.hpp`)

- ``select.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_decision_tree_all_dimension_select.hpp`)

- ``sequential.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_sequential.hpp`)

- ``softmax.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_log_softmax.hpp`)

- ``softmin.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_softmin.hpp`)

- ``softshrink.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_softshrink.hpp`)

- ``spatial_dropout.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_spatial_dropout.hpp`)

- ``subview.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_subview.hpp`)

- ``transposed_convolution.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_transposed_convolution.hpp`)

- ``virtual_batch_norm.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_virtual_batch_norm.hpp`)

- ``vr_class_reward.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_vr_class_reward.hpp`)

- ``weight_norm.hpp`` (:ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_weight_norm.hpp`)



Included By
-----------


- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_brnn.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_ffn.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_layer_names.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_ann_rnn.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_networks_categorical_dqn.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_networks_dueling_dqn.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_networks_simple_dqn.hpp`

- :ref:`file__home_aakash_mlpack_src_mlpack_tests_custom_layer.hpp`




Full File Listing
-----------------

.. doxygenfile:: /home/aakash/mlpack/src/mlpack/methods/ann/layer/layer.hpp

