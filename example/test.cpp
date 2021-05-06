#include "cudaSolver.h"
#include <iostream>
#include <vector>
#include <ctime>

const int MESH = 1000;
const int SCALE = MESH * MESH;

int main()
{
    Eigen::SparseMatrix<Cuda::ValueType> A(SCALE, SCALE);
    Eigen::Matrix<Cuda::ValueType, Eigen::Dynamic, 1> b(SCALE), x(SCALE);
    Eigen::Matrix<Cuda::ValueType, Eigen::Dynamic, 1> ones = Eigen::Matrix<Cuda::ValueType, Eigen::Dynamic, 1>::Ones(SCALE);

    int mesh_val_num = 3 * MESH - 2;
    int total_val_num = MESH * mesh_val_num + 2 * (SCALE - MESH);
    std::vector<Eigen::Triplet<Cuda::ValueType>> tripList;
    for (int i = 0; i < MESH; i++) {
        for (int j = 0; j < MESH; j++) {
           tripList.push_back(Eigen::Triplet<Cuda::ValueType>(MESH * i + j, MESH * i + j, 4));
        }
        for (int j = MESH; j < 2 * MESH - 1; j++) {
            tripList.push_back(Eigen::Triplet<Cuda::ValueType>(MESH * i + j % MESH, MESH * i + (j + 1) % MESH, -1));
        }
        for (int j = 2 * MESH - 1; j < 3 * MESH - 2; j++) {
            tripList.push_back(Eigen::Triplet<Cuda::ValueType>(MESH * i + (j + 2) % MESH, MESH * i + (j + 1) % MESH, -1));
        }
    }
    for (int i = mesh_val_num * MESH; i < (total_val_num - (SCALE - MESH)); i++) {
        tripList.push_back(Eigen::Triplet<Cuda::ValueType>(i - mesh_val_num*MESH, i - mesh_val_num*MESH + MESH, -1));
    }
    for (int i = total_val_num - (SCALE - MESH); i < total_val_num; i++) {
        tripList.push_back(Eigen::Triplet<Cuda::ValueType>(i - (total_val_num - (SCALE - MESH)) + MESH, i - (total_val_num - (SCALE - MESH)), -1));
    }

    A.setFromTriplets(tripList.begin(), tripList.end());
    b = A * ones;
    
    std::clock_t start, end;

    start = clock();
    
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Cuda::ValueType>, Eigen::Lower|Eigen::Upper> cg;
    cg.compute(A);
    x = cg.solve(b);
    std::cout << "#iterations: " << cg.iterations() << std::endl;
    
    end = clock();

    std::cout << "CG by Eigen: " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    start = clock();
   
    Cuda::CudaSolver(A, x, b);
    
    /*
    for (int i = 0; i < x.size(); i++) {
        std::cout << x(i) << std::endl;
    }
    */
    

    end = clock();
    std::cout << "CG by AMGX: " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    return 0;
}