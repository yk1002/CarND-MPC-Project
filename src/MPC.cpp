#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"
#include "myutil.hpp"

using namespace std;
using CppAD::AD;

/*
  (N, dt) experiment results:

  N   dt    Result
  ----------------------------------------------------------
  5   0.05  fail
  10  0.05  fail
  20  0.05  fail
  5   0.1   fail
  10  0.1   success; fastest wrap
  20  0.1   success; slows down at corners
  5   0.2   failure
  10  0.2   success
  20  0.2   success; slows down at corners
 */

const size_t N = 10;
const size_t N_control = N - 1;
const double dt = 0.1;
const double Lf = 2.67;
const double ref_v = 100;
const size_t n_state_vars = 6 * N;
const size_t n_control_vars = 2 * N_control;
const size_t n_vars = n_state_vars + n_control_vars;
const size_t n_constraints = N * 6;

const size_t x_start = 0;
const size_t y_start = x_start + N;
const size_t psi_start = y_start + N;
const size_t v_start = psi_start + N;
const size_t cte_start = v_start + N;
const size_t epsi_start = cte_start + N;
const size_t delta_start = epsi_start + N;
const size_t a_start = delta_start + N_control;

template <typename T>
void fill(T& vec, size_t begin, size_t end, double v) {
    for (auto i = begin; i < end; ++i) vec[i] = v;
}

struct FG_eval {
    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

    Eigen::VectorXd coeffs, d_coeffs;
    
    FG_eval(const Eigen::VectorXd& coeffs) : coeffs(coeffs), d_coeffs(get_derivative_coeffs(coeffs)) {
        assert(coeffs.size() == 4); // expecting 3 degree polyminial
    }

    void operator()(ADvector& fg, const ADvector& vars) {
        //
        // Setting up the objective function to minimize.
        //
        
        // fg[0] is the objective function the solver will minimize.
        auto& objf = fg[0];
        objf = 0.0; 
      
        // These weights are taken from the youtube video "Self-Driving Car Project Q&A | MPC Controller."
        // https://www.youtube.com/watch?v=bOQuhpz3YfU&index=5&list=PLAwxTw4SYaPnfR7TzRZN-uxlxGbqxhtm2
        const double track_weight = 2000;
        const double speed_weight = 1;
        const double acutuator_usage_weight = 5;
        const double steering_smoothness_weight = 200;
        const double throttle_smoothness_weight = 10;

        // The part of the cost based on the reference state.
        for (int t = 0; t < N; t++) {
            objf += CppAD::pow(vars[cte_start + t], 2) * track_weight;
            objf += CppAD::pow(vars[epsi_start + t], 2) * track_weight;
            objf += CppAD::pow(vars[v_start + t] - ref_v, 2) * speed_weight;
        }
      
        // Minimize the use of actuators.
        for (int t = 0; t < N_control; ++t) {
            objf += CppAD::pow(vars[delta_start + t], 2) * acutuator_usage_weight;
            objf += CppAD::pow(vars[a_start + t], 2) * acutuator_usage_weight;
        }
        
        // Minimize the value gap between sequential actuations.
        for (int t = 0; t < N_control - 1; ++t) {
            objf += CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2) * steering_smoothness_weight;
            objf += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2) * throttle_smoothness_weight;
        }
        
        //
        // Setting up constraints
        //

        // states at t = 0;
        int t = 0;
        fg[1 + x_start + t] = vars[x_start + t];
        fg[1 + y_start + t] = vars[y_start + t];
        fg[1 + psi_start + t] = vars[psi_start + t];
        fg[1 + v_start + t] = vars[v_start + t];
        fg[1 + cte_start + t] = vars[cte_start + t];
        fg[1 + epsi_start + t] = vars[epsi_start + t];
      
        // constraints for future state transitions
        for (t = 1; t < N; ++t) {
            // The state at time t+1 .
            AD<double> x1 = vars[x_start + t];
            AD<double> y1 = vars[y_start + t];
            AD<double> psi1 = vars[psi_start + t];
            AD<double> v1 = vars[v_start + t];
            AD<double> cte1 = vars[cte_start + t];
            AD<double> epsi1 = vars[epsi_start + t];
            
            // The state at time t.
            AD<double> x0 = vars[x_start + t - 1];
            AD<double> y0 = vars[y_start + t - 1];
            AD<double> psi0 = vars[psi_start + t - 1];
            AD<double> v0 = vars[v_start + t - 1];
            AD<double> cte0 = vars[cte_start + t - 1];
            AD<double> epsi0 = vars[epsi_start + t - 1];
            
            // Only consider the actuation at time t.
            AD<double> delta0 = vars[delta_start + t - 1];
            AD<double> a0 = vars[a_start + t - 1];
            
            AD<double> f0 = 0.0;
            for (int i = 0; i < coeffs.size(); ++i)
                f0 += coeffs[i] * CppAD::pow(x0, i);
            AD<double> d_f0 = 0.0;
            for (int i = 0; i < d_coeffs.size(); ++i)
                d_f0 += d_coeffs[i] * CppAD::pow(x0, i);
            AD<double> psides0 = CppAD::atan(d_f0);
            
            fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
            fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
            fg[1 + psi_start + t] = psi1 - (psi0 + v0 / Lf * -delta0 * dt); // flipping delta to match simulator
            fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
            fg[1 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
            fg[1 + epsi_start + t] = epsi1 - ((psi0 - psides0) + v0 * -delta0 / Lf * dt);// flipping delta to match simulator
        }
    }
};

//
// MPC class definition implementation.
//
MPC::MPC() : x(N), y(N), steering(N_control), throttle(N_control) {}

void MPC::solve(const Eigen::VectorXd& state, const Eigen::VectorXd& coeffs) {
    typedef CPPAD_TESTVECTOR(double) Dvector;

    const double x0 = state[0];
    const double y0 = state[1];
    const double psi0 = state[2];
    const double v0 = state[3];
    const double cte0 = state[4];
    const double epsi0 = state[5];

    // Initial value of the independent variables set to 0 except for t = 0
    Dvector vars(n_vars);
    fill(vars, 0, n_vars, 0.0);
    vars[x_start] = x0;
    vars[y_start] = y0;
    vars[psi_start] = psi0;
    vars[v_start] = v0;
    vars[cte_start] = cte0;
    vars[epsi_start] = epsi0;

    // Lower and upper limits for state variables
    Dvector vars_lowerbound(n_vars);
    Dvector vars_upperbound(n_vars);
    fill(vars_lowerbound, 0, n_state_vars, -1.0e19);
    fill(vars_upperbound, 0, n_state_vars, 1.0e19);
    fill(vars_lowerbound, delta_start, delta_start + N_control, -0.436332); // delta (steering); -25 degrees in radians
    fill(vars_upperbound, delta_start, delta_start + N_control, 0.436332);  // delta (steering); 25 degreess in radians
    fill(vars_lowerbound, a_start, a_start + N_control, -1.0); // acceleration
    fill(vars_upperbound, a_start, a_start + N_control, 1.0);  // acceleration

    // Lower and upper limits for the constraints should be 0 besides initial state.
    Dvector constraints_lowerbound(n_constraints);
    Dvector constraints_upperbound(n_constraints);
    fill(constraints_lowerbound, 0, n_constraints, 0.0);
    fill(constraints_upperbound, 0, n_constraints, 0.0);

    constraints_lowerbound[x_start] = x0;
    constraints_lowerbound[y_start] = y0;
    constraints_lowerbound[psi_start] = psi0;
    constraints_lowerbound[v_start] = v0;
    constraints_lowerbound[cte_start] = cte0;
    constraints_lowerbound[epsi_start] = epsi0;
    
    constraints_upperbound[x_start] = x0;
    constraints_upperbound[y_start] = y0;
    constraints_upperbound[psi_start] = psi0;
    constraints_upperbound[v_start] = v0;
    constraints_upperbound[cte_start] = cte0;
    constraints_upperbound[epsi_start] = epsi0;

    // object that computes objective and constraints
    FG_eval fg_eval(coeffs);

    // options for IPOPT solver
    std::string options;
    options += "Integer print_level  0\n";
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    options += "Numeric max_cpu_time          0.5\n";

    // solve the problem
    CppAD::ipopt::solve_result<Dvector> solution;
    CppAD::ipopt::solve<Dvector, FG_eval>(
        options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
        constraints_upperbound, fg_eval, solution);

    // Check some of the solution values
    assert(solution.status == CppAD::ipopt::solve_result<Dvector>::success);

    // Filling calculated state and control values
    cost = solution.obj_value;

    for (int t = 0; t < N; ++t) {
        x[t] = solution.x[x_start + t];
        y[t] = solution.x[y_start + t];
    }

    for (int t = 0; t < N_control; ++t) {
        steering[t] = solution.x[delta_start + t];
        throttle[t] = solution.x[a_start + t];
    }
}
