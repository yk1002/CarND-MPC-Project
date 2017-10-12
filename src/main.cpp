#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/LU"
#include "MPC.h"
#include "json.hpp"
#include "myutil.hpp"

using namespace std;

bool verbose = false;

#define V(v) #v": "<< v

template <typename T>
ostream& print(ostream& os, const T& vec) {
    bool first = true;
    os << '{';
    for (int i = 0; i < vec.size(); ++i) {
        auto& v = vec[i];
        if (!first) 
            os << ", ";
        first = false;
        os << v;
    }
    os << '}';
    return os;
}

ostream& operator << (ostream& os, const vector<double>& vec) { return print(os, vec); }
ostream& operator << (ostream& os, const Eigen::VectorXd& vec) { return print(os, vec); }

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(const Eigen::VectorXd& coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(const Eigen::VectorXd& xvals, const Eigen::VectorXd& yvals, int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

/* Convert points in the global frame to the car's local frame
   represented by the tuple (px, py, psi) by multiplying to those
   points the inverse matrix of the local-to-global matrix used in
   the kidnapped car project.
   
   local to global conversion:
   | Xg |    | cos(psi) -sin(psi) px | | Xc |
   | Yg | =  | sin(psi)  cos(psi) py | | Yc |
   | 1  |    |        0         0  1 | |  1 |
   
   global to local conversion (inverse of the matrix above):
   | Xc |    |  cos(psi)  sin(psi) -sin(psi)*py-cos(psi)*px | | Xg |
   | Yc | =  | -sin(psi)  cos(psi)  sin(psi)*px-cos(psi)*py | | Yg |
   | 1  |    |         0         0                        1 | |  1 |
*/
Eigen::Matrix3d global_to_local_matrix(double px, double py, double psi) {
    const auto c = cos(psi);
    const auto s = sin(psi);
    Eigen::Matrix3d global_to_local;
    global_to_local << 
        c,     s, -s * py - c * px,
        -s,    c,  s * px - c * py,
        0.0, 0.0,              1.0;
    return global_to_local;
}

template <typename T1, typename T2>
void change_frame(const Eigen::Matrix3d& conv, const T1& ptsx_global, const T1& ptsy_global, T2& ptsx_local, T2& ptsy_local) {
    const int N_pts = ptsx_global.size();
    assert(N_pts == ptsx_local.size());
    assert(N_pts == ptsy_local.size());
    for (int i = 0; i < N_pts; ++i) {
        const Eigen::Vector3d pts_global(ptsx_global[i], ptsy_global[i], 1.0);
        const auto pts_local = conv * pts_global;
        ptsx_local[i] = pts_local[0];
        ptsy_local[i] = pts_local[1];
    }
}

void interactive_coodinate_conversion()
{
    cout << "Interactive coodinate conversion:" << endl;
    double px, py, psi;
    cout << "Specify px, py and psi (in degrees): ";
    cin >> px >> py >> psi;
    cout.precision(3);

    const Eigen::Matrix3d global_to_local = global_to_local_matrix(px, py, deg2rad(psi));
    cout << "Global to Local matrix:" << endl << global_to_local << endl;
    cout << endl;

    double gx, gy;
    while (cin >> gx >> gy) {
        const Eigen::Vector3d pts_g(gx, gy, 1.0);
        const auto pts_l = global_to_local * pts_g;
        cout << "Global coordinate = (" << gx << ", " << gy << ")" << endl;
        cout << "Local  coordinate = (" << pts_l[0] << ", " << pts_l[0] << ")" << endl;
    }
}

int main(int argc, char** argv) {
  uWS::Hub h;

  if (argc > 1 && string(argv[1]) == "interactive_coodinate_conversion") {
      interactive_coodinate_conversion();
      return 0;
  }

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          const vector<double> ptsx = j[1]["ptsx"];
          const vector<double> ptsy = j[1]["ptsy"];
          const double delta = j[1]["steering_angle"];
          const double acceleration = j[1]["throttle"];
          double x = j[1]["x"];
          double y = j[1]["y"];
          double psi = double(j[1]["psi"]);
          double v = j[1]["speed"];

          const int N_pts = ptsx.size();
          if (verbose) {
              cout << "============================================================" << endl;
              cout << V(N_pts) << endl;
              cout << V(ptsx) << endl;
              cout << V(ptsy) << endl;
              cout << V(x) << endl;
              cout << V(y) << endl;
              cout << V(psi) << endl;
              cout << V(rad2deg(psi)) << endl;
              cout << V(v) << endl;
          }

          // take latency into account by advancing states by the same amount as latency
          const double latency = 0.1; // 100ms
          const double Lf = 2.67;
          x += v * cos(psi) * latency;
          y +=v * sin(psi) * latency;
          psi += v * -delta / Lf * latency; // flipping delta to match simulator
          v += acceleration * latency;

          // get matrix that converts global coordinates to local ones
          const auto mat = global_to_local_matrix(x, y, psi);

          // convert way points to local frame
          Eigen::VectorXd ptsx_local(N_pts), ptsy_local(N_pts);
          change_frame(mat, ptsx, ptsy, ptsx_local, ptsy_local);
          if (verbose) {
              cout << V(ptsx_local) << endl;
              cout << V(ptsy_local) << endl;
          }

          // fit 3 degree polynomial to way points
          const auto coeffs = polyfit(ptsx_local, ptsy_local, 3);

          // set x, y, psi, cte, and epsi in local frame
          const auto x_local = 0.0, y_local = 0.0, psi_local = 0.0;
          const auto cte = polyeval(coeffs, x_local) - y_local;
          const auto epsi = psi_local - atan(polyeval(get_derivative_coeffs(coeffs), x_local));

          // solve it
          Eigen::VectorXd state(6);
          state << x_local, y_local, psi_local, v, cte, epsi;
          mpc.solve(state, coeffs);

          json msgJson;
          msgJson["steering_angle"] = mpc.steering[0];
          msgJson["throttle"] = mpc.throttle[0];

          // Display the MPC predicted trajectory 
          msgJson["mpc_x"] = mpc.x;
          msgJson["mpc_y"] = mpc.y;

          // Display the waypoints/reference line in yellow
          vector<double> next_x_vals(N_pts);
          vector<double> next_y_vals(N_pts);
          change_frame(mat, ptsx, ptsy, next_x_vals, next_y_vals);
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
