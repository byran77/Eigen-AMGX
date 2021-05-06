#pragma once

#include <vector>
#include <memory>

namespace Cuda {

using ValueType = double;

class CRS {
public:
    CRS(size_t rows, size_t cols, size_t nonzeros);
    ~CRS();

    int* innerIndexPtr();
    int* outerIndexPtr();
    ValueType* valuePtr();

    size_t innerSize();
    size_t outerSize();
    size_t nonZeros();

private:
    int* m_ptr_inner;
    int* m_ptr_outer;
    ValueType* m_ptr_value;
    size_t m_rows;
    size_t m_cols;
    size_t m_nonzeros;
};

class vec {
public:
    vec(size_t size);
    ~vec();

    size_t size();
    ValueType* valuePtr();

private:
    ValueType* m_ptr_value;
    size_t m_size;
};

using CRS_ptr = std::shared_ptr<CRS>;
using vec_ptr = std::shared_ptr<vec>;

void cudaSolver(CRS_ptr A, vec_ptr x, vec_ptr b);

}