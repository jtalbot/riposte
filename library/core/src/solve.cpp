
#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

#include "../../../libs/Eigen/Dense"

extern "C"
Value solve(Thread& thread, Value const* args) {
    int64_t As = ((Integer const&)args[1])[0];
    Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>((double*)((Double const&)args[0]).v(), As, As);
    int64_t Br = ((Integer const&)args[3])[0];
    int64_t Bc = ((Integer const&)args[4])[0];
    Eigen::MatrixXd B = Eigen::Map<Eigen::MatrixXd>((double*)((Double const&)args[2]).v(), Br, Bc);

    Eigen::FullPivLU<Eigen::MatrixXd> EA(A);
    Double result(As*Bc);
    Eigen::Map<Eigen::MatrixXd>(result.v(), As, Bc) = EA.solve(B);
    return result;
}

