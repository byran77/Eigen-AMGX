#pragma once

#include "transMatrix.h"
#include "Eigen/Sparse"

namespace Cuda {

void CudaSolver(const Eigen::SparseMatrix<ValueType>& A, Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& x, const Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& b);

CRS_ptr SparseMatrix2CRS(const Eigen::SparseMatrix<ValueType>& matrix);
vec_ptr Vector2Ptr(const Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& vector);
void Ptr2Vector(vec_ptr ptr, Eigen::Matrix<ValueType, Eigen::Dynamic, 1>& vector);
}