#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

struct MPC {
    MPC();
    void solve(const Eigen::VectorXd& state, const Eigen::VectorXd& coeffs);
    double cost;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> steering;
    std::vector<double> throttle;
};

#endif /* MPC_H */
