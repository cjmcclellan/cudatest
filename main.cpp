#include <iostream>
#include "helper.h"
#include "loadmodels_incuda.h"

int main() {
    cublasHandle_t cublasHandle;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
//    cublasStatus = cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
    checkCublasErrors(cublasStatus);
    std::cout << "Hello, World!" << std::endl;
    runModel();
    return 0;
}
