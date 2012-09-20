/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef NVVM_H
#define NVVM_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>

/*
 * nvvmCU
 * - an opaque handle for compilation unit
 */
typedef struct _nvvmCU *nvvmCU;


/*
 * NVVM API call result code
 */
typedef enum {
    NVVM_SUCCESS = 0,
    NVVM_ERROR_OUT_OF_MEMORY,
    NVVM_ERROR_NOT_INITIALIZED,
    NVVM_ERROR_ALREADY_INITIALIZED,
    NVVM_ERROR_CU_CREATION_FAILURE,
    NVVM_ERROR_IR_VERSION_MISMATCH,
    NVVM_ERROR_INVALID_INPUT,
    NVVM_ERROR_INVALID_CU,
    NVVM_ERROR_INVALID_IR,
    NVVM_ERROR_INVALID_OPTION,
    NVVM_ERROR_NO_MODULE_IN_CU,
    NVVM_ERROR_COMPILATION,
} nvvmResult;


/*
 * Init and fini 
 *
 * nvvmInit() must be called before any of the other libnvvm API functions 
 * can be called. 
 * 
 * Once nvvmFini() is called, no other libnvvm API functions (except for
 * nvvmInit() can be called).
 */
nvvmResult nvvmInit();
nvvmResult nvvmFini();


/*
 * Get the libNVVM version
 *
 * Return value:
 *    NVVM_SUCCESS,
 *    NVVM_ERROR_NOT_INITIALIZED,
 */
nvvmResult nvvmVersion(int *major, int *minor);


/*
 * Create a compilation unit, and set the value of its handle to *cu.
 *
 * Return value:
 *    NVVM_SUCCESS,
 *    NVVM_ERROR_OUT_OF_MEMORY,
 *    NVVM_ERROR_NOT_INITIALIZED,
 */
nvvmResult nvvmCreateCU(nvvmCU *cu);


/*
 * Destroy a compilation unit
 *
 * Return value:
 *    NVVM_SUCCESS,
 *    NVVM_ERROR_NOT_INITIALIZED,
 */
nvvmResult nvvmDestroyCU(nvvmCU *cu);


/*
 * Add a module level NVVM IR to a compilation unit.
 * - The buffer should contain an NVVM module IR either in the bitcode
 *   representation or in the text representation.
 *
 * Return value:
 *    NVVM_SUCCESS,
 *    NVVM_ERROR_OUT_OF_MEMORY,
 *    NVVM_ERROR_NOT_INITIALIZED,
 *    NVVM_ERROR_INVALID_INPUT,
 *    NVVM_ERROR_INVALID_CU,
 */
nvvmResult nvvmCUAddModule(nvvmCU cu, const char *buffer, size_t size);


/*
 * Perform Compliation
 *
 * The valid compiler options are
 *
 * -target=<value>
 *     <value>: ptx (default), verify
 * -g
 * -opt=<level>
 *     <level>: 0, 3 (default)
 * -arch=<arch_value>
 *     <arch_value>: compute_20 (default), compute_30
 * -ftz=<value>
 *     <value>: 0 (default, preserve denormal values, when performing
 *                 single-precision floating-point operations)
 *              1 (flush denormal values to zero, when performing
 *                 single-precision floating-point operations)
 * -prec-sqrt=<value>
 *     <value>: 0 (use a faster approximation for single-precision
 *                 floating-point square root)
 *              1 (default, use IEEE round-to-nearest mode for
 *                 single-precision floating-point square root)
 * -prec-div=<value>
 *     <value>: 0 (use a faster approximation for single-precision
 *                 floating-point division and reciprocals)
 *              1 (default, use IEEE round-to-nearest mode for
 *                 single-precision floating-point division and reciprocals)
 * -fma=<value>
 *     <value>: 0 (disable FMA contraction),
 *              1 (default, enable FMA contraction),
 *
 * Return value:
 *    NVVM_SUCCESS,
 *    NVVM_ERROR_OUT_OF_MEMORY,
 *    NVVM_ERROR_NOT_INITIALIZED,
 *    NVVM_ERROR_IR_VERSION_MISMATCH,
 *    NVVM_ERROR_INVALID_CU,
 *    NVVM_ERROR_INVALID_IR,
 *    NVVM_ERROR_INVALID_OPTION,
 *    NVVM_ERROR_NO_MODULE_IN_CU,
 *    NVVM_ERROR_COMPILATION,
 */
nvvmResult nvvmCompileCU(nvvmCU cu, int numOptions, const char **options);   


/*
 * Get the size of the compiled result
 *
 * Return value:
 *    NVVM_SUCCESS,
 *    NVVM_ERROR_NOT_INITIALIZED,
 *    NVVM_ERROR_INVALID_CU,
 */
nvvmResult nvvmGetCompiledResultSize(nvvmCU cu, size_t *bufferSizeRet);


/*
 * Get the compiled result
 * - the result is stored in the memory pointed by 'buffer'
 *
 * Return value:
 *    NVVM_SUCCESS,
 *    NVVM_ERROR_NOT_INITIALIZED,
 *    NVVM_ERROR_INVALID_CU,
 */
nvvmResult nvvmGetCompiledResult(nvvmCU cu, char *buffer);



/*
 * Get the Size of Compiler Message
 * - The size of the message string (including the trailing
 *   NULL) is stored into 'buffer_size_ret' when the return
 *   value is NVVM_SUCCESS.
 *   
 * Return value:
 *    NVVM_SUCCESS,
 *    NVVM_ERROR_NOT_INITIALIZED,
 *    NVVM_ERROR_INVALID_CU,
 */
nvvmResult nvvmGetCompilationLogSize(nvvmCU cu, size_t *bufferSizeRet);


/*
 * Get the Compiler Message
 * - The NULL terminated message string is stored in the
 *   memory pointed by 'buffer' when the return value is
 *   NVVM_SUCCESS.
 *   
 * Return value:
 *    NVVM_SUCCESS,
 *    NVVM_ERROR_NOT_INITIALIZED,
 *    NVVM_ERROR_INVALID_CU,
 */
nvvmResult nvvmGetCompilationLog(nvvmCU cu, char *buffer);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* NVVM_H */
