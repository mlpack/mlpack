/**
 * @file methods/ann/gan/gan_policies.hpp
 * @author Shikhar Jaiswal
 *
 * Implementation of the GAN policy types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_GAN_POLICIES_HPP
#define MLPACK_METHODS_ANN_GAN_GAN_POLICIES_HPP

namespace mlpack {

/**
 * For more information, see the following paper:
 *
 * @code
 * @article{Salimans16,
 *   author    = {Tim Salimans, Ian Goodfellow, Wojciech Zaremba,
 *                Vicki Cheung, Alec Radford and Xi Chen},
 *   title     = {Improved Techniques for Training GANs},
 *   year      = {2016},
 *   url       = {http://arxiv.org/abs/1606.03498},
 *   eprint    = {1606.03498},
 * }
 * @endcode
 */
class StandardGAN { /* Nothing to do here */ };

/**
 * For more information, see the following paper:
 *
 * @code
 * @article{Radford15,
 *   author    = {Alec Radford, Luke Metz and Soumith Chintala},
 *   title     = {Unsupervised Representation Learning with Deep Convolutional
                  Generative Adversarial Networks},
 *   year      = {2015},
 *   url       = {https://arxiv.org/abs/1511.06434},
 *   eprint    = {1511.06434},
 * }
 * @endcode
 */
class DCGAN { /* Nothing to do here */ };

/**
 * For more information, see the following paper:
 *
 * @code
 * @article{Arjovsky17,
 *   author    = {Martin Arjovsky, Soumith Chintala and Léon Bottou},
 *   title     = {Wasserstein GAN},
 *   year      = {2017},
 *   url       = {https://arxiv.org/abs/1701.07875},
 *   eprint    = {1701.07875},
 * }
 * @endcode
 */
class WGAN { /* Nothing to do here */ };

/**
 * For more information, see the following paper:
 *
 * @code
 * @article{Gulrajani17,
 *   author    = {Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent
                  Dumoulin and Aaron Courville},
 *   title     = {Improved Training of Wasserstein GANs},
 *   year      = {2017},
 *   url       = {https://arxiv.org/abs/1704.00028},
 *   eprint    = {1704.00028},
 * }
 * @endcode
 */
class WGANGP { /* Nothing to do here */ };

} // namespace mlpack

#endif
