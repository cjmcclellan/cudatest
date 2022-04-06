//
// Created by connor on 4/6/22.
//

#ifndef CUDATEST_HELPER_H
#define CUDATEST_HELPER_H
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"
#include "cuda.h"

#define checkCublasErrors(err) __checkCublasErrors(err, __FILE__, __LINE__)

static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

// These are the inline versions for all of the SDK helper functions
inline void __checkCublasErrors(cublasStatus_t err, const char *file, const int line) {
    if (CUBLAS_STATUS_SUCCESS != err) {
        const char *errorStr = NULL;
        errorStr = _cublasGetErrorEnum(err);
        fprintf(stderr,
                "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
                "line %i.\n",
                err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif //CUDATEST_HELPER_H
