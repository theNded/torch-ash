/*
 *  Copyright 2019 Patrick Stotko
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef STDGPU_CONFIG_H
//! @cond Doxygen_Suppress
#define STDGPU_CONFIG_H
//! @endcond

/**
 * \addtogroup config config
 * \ingroup system
 * @{
 */

/**
 * \file stdgpu/config.h.in
 * \brief This file serves as a template for CMake which configures the corresponding config.h file.
 */



/**
 * \brief Namespace containing all defined classes and functions
 */
namespace stdgpu
{

/**
 * \ingroup config
 * \brief Major version component of stdgpu library
 */
#define STDGPU_VERSION_MAJOR 1
/**
 * \ingroup config
 * \brief Minor version component of stdgpu library
 */
#define STDGPU_VERSION_MINOR 3
/**
 * \ingroup config
 * \brief Patch version component of stdgpu library
 */
#define STDGPU_VERSION_PATCH 0
/**
 * \ingroup config
 * \brief Version string of stdgpu library
 */
#define STDGPU_VERSION_STRING "1.3.0"


//! @cond Doxygen_Suppress
#define STDGPU_BACKEND STDGPU_BACKEND_CUDA
#define STDGPU_BACKEND_DIRECTORY cuda
#define STDGPU_BACKEND_NAMESPACE cuda
#define STDGPU_BACKEND_MACRO_NAMESPACE CUDA
//! @endcond


/**
 * \ingroup config
 * \def STDGPU_ENABLE_CONTRACT_CHECKS
 * \brief Library option to enable contract checks
 */
// Workaround: Provide a define only for the purpose of creating the documentation since Doxygen does not recognize #cmakedefine01
#ifdef STDGPU_RUN_DOXYGEN
    #define STDGPU_ENABLE_CONTRACT_CHECKS
#endif
#define STDGPU_ENABLE_CONTRACT_CHECKS 1

/**
 * \ingroup config
 * \def STDGPU_USE_32_BIT_INDEX
 * \brief Library option to use 32-bit integers rather than 64-bit to define index_t
 */
// Workaround: Provide a define only for the purpose of creating the documentation since Doxygen does not recognize #cmakedefine01
#ifdef STDGPU_RUN_DOXYGEN
    #define STDGPU_USE_32_BIT_INDEX
#endif
#define STDGPU_USE_32_BIT_INDEX 1

} // namespace stdgpu



/**
 * @}
 */



#endif // STDGPU_CONFIG_H
