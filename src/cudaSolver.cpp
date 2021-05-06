#include "cudaSolver.h"
#include <cassert>
#include <cstring>

namespace Cuda {

void CudaSolver(const Eigen::SparseMatrix<ValueType>& para_A, Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& para_x, const Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& para_b)
{
    assert(para_x.size() == para_b.size());
    CRS_ptr A = SparseMatrix2CRS(para_A);
    vec_ptr b = Vector2Ptr(para_b), x = Vector2Ptr(para_x);

    cudaSolver(A, x, b);

    Ptr2Vector(x, para_x);

    return ;
}


CRS_ptr SparseMatrix2CRS(const Eigen::SparseMatrix<ValueType>& matrix)
{
    Eigen::SparseMatrix<ValueType, Eigen::RowMajor> crs_matrix(matrix);
    crs_matrix.makeCompressed();
    int rows = crs_matrix.rows(), cols = crs_matrix.cols(), nonZeros = crs_matrix.nonZeros();
    CRS_ptr mat_ptr = std::make_shared<CRS>(rows, cols, nonZeros);
    
    int* inner_ptr = crs_matrix.innerIndexPtr(), * outer_ptr = crs_matrix.outerIndexPtr();
    ValueType* value_ptr = crs_matrix.valuePtr();
    memcpy(mat_ptr->valuePtr(), value_ptr, sizeof(ValueType) * nonZeros);
    memcpy(mat_ptr->innerIndexPtr(), inner_ptr, sizeof(int) * nonZeros);
    memcpy(mat_ptr->outerIndexPtr(), outer_ptr, sizeof(int) * (mat_ptr->outerSize() + 1));

    return mat_ptr;
}

vec_ptr Vector2Ptr(const Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& vector)
{
    size_t n = vector.size();
    vec_ptr ptr = std::make_shared<vec>(n);
    memcpy(ptr->valuePtr(), vector.data(), sizeof(ValueType) * n);

    return ptr;
}

void Ptr2Vector(vec_ptr ptr, Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& vector)
{
    size_t n = ptr->size();
    assert(vector.size() == n);

    memcpy(vector.data(), ptr->valuePtr(), sizeof(ValueType) * n);

    return ;
}

}