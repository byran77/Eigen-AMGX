#include "transMatrix.h"
#include "amgx_c.h"
#include <cassert>
//#include <sstream>

namespace Cuda {

void cudaSolver(CRS_ptr para_A, vec_ptr para_x, vec_ptr para_b)
{
    assert(para_A->innerSize() == para_b->size() && para_x->size() == para_b->size());

    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());

    /*
    std::stringstream conf_stream;
    conf_stream << "config_version=2, " << "solver(cg)=CG, " << "cg:preconditioner=NOSOLVER, " \
                << "cg:max_iters=1000, " <<"cg:tolerance=1e-6, " << "cg:monitor_residual=1, " << "cg:print_solve_stats=1";
    */

    AMGX_matrix_handle A;
    AMGX_vector_handle b;
    AMGX_vector_handle x;
    AMGX_resources_handle rsrc;
    AMGX_solver_handle solver;
    AMGX_config_handle cfg;

    //AMGX_config_create(&cfg, conf_stream.str().c_str());
    AMGX_config_create_from_file(&cfg, "E:\\AMGX_configs_xiao\\531config.json");
    AMGX_resources_create_simple(&rsrc, cfg);
    AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI);
    AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI);
    AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg);

    AMGX_pin_memory(para_A->outerIndexPtr(), sizeof(int) * (para_A->outerSize() + 1));
    AMGX_pin_memory(para_A->innerIndexPtr(), sizeof(int) * (para_A->nonZeros()));
    AMGX_pin_memory(para_A->valuePtr(), sizeof(ValueType) * (para_A->nonZeros()));
    AMGX_pin_memory(para_b->valuePtr(), sizeof(ValueType) * (para_b->size()));
    AMGX_pin_memory(para_x->valuePtr(), sizeof(ValueType) * (para_x->size()));

    AMGX_matrix_upload_all(A, para_A->outerSize(), para_A->nonZeros(), 1, 1, para_A->outerIndexPtr(), para_A->innerIndexPtr(), para_A->valuePtr(), NULL);
    AMGX_vector_upload(b, para_b->size(), 1, para_b->valuePtr());
    AMGX_vector_set_zero(x, para_x->size(), 1);

    AMGX_solver_setup(solver, A);
    AMGX_solver_solve_with_0_initial_guess(solver, b, x);

    AMGX_vector_download(x, para_x->valuePtr());

    AMGX_unpin_memory(para_A->outerIndexPtr());
    AMGX_unpin_memory(para_A->innerIndexPtr());
    AMGX_unpin_memory(para_A->valuePtr());
    AMGX_unpin_memory(para_b->valuePtr());
    AMGX_unpin_memory(para_x->valuePtr());

    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());

    return ;
}

}