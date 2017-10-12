#ifndef MYUTIL_H
#define MYUTIL_H

#include <algorithm>
#include <cassert>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/LU"

inline Eigen::VectorXd get_derivative_coeffs(const Eigen::VectorXd& coeffs) {
    Eigen::VectorXd d_coeffs(std::max(1, int(coeffs.size()) - 1));
    d_coeffs[0] = 0.0;
    for (int i = 0; i < d_coeffs.size(); ++i)
        d_coeffs[i] = (i + 1) * coeffs[i + 1];
    return d_coeffs;
}

#endif
