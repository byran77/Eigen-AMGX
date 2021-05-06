#include "transMatrix.h"

namespace Cuda {

CRS::CRS(size_t rows, size_t cols, size_t nonzeros): \
m_rows(rows), m_cols(cols), m_nonzeros(nonzeros)
{
    m_ptr_value = new ValueType[nonzeros];
    m_ptr_inner = new int[nonzeros];
    m_ptr_outer = new int[rows + 1];
}

CRS::~CRS()
{
    delete[] m_ptr_value;
    delete[] m_ptr_inner;
    delete[] m_ptr_outer;
}

int* CRS::innerIndexPtr()
{
    return m_ptr_inner;
}

int* CRS::outerIndexPtr()
{
    return m_ptr_outer;
}

ValueType* CRS::valuePtr()
{
    return m_ptr_value;
}

size_t CRS::innerSize()
{
    return m_cols;
}

size_t CRS::outerSize()
{
    return m_rows;
}

size_t CRS::nonZeros()
{
    return m_nonzeros;
}

vec::vec(size_t size): m_size(size)
{
    m_ptr_value = new ValueType[size];
}

vec::~vec()
{
    delete[] m_ptr_value;
}

size_t vec::size()
{
    return m_size;
}

ValueType* vec::valuePtr()
{
    return m_ptr_value;
}


}