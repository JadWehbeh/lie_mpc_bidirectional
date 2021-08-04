# include <optimization.h>
# include <Eigen/Dense>
# include <math.h>
# include <iostream>


namespace al = alglib;

class LieMPC {

  private:
    int _nx = 12;                                             // Number of states
    int _ny = 6;                                              // Number of outputs
    int _nu = 4;                                              // Number of control inputs
    int _np;                                                  // Prediction steps
    double _ts;                                               // Time step (s)
    double _T_max;                                            // Maximum thrust (N)
    double _t_ramp;                                           // Thrust ramp time (s)
    Eigen::MatrixXd _J = Eigen::MatrixXd::Zero(3,3);          // Inertia matrix (kg.m^2)
    Eigen::MatrixXd _J_inv;                                   // Inverse of inertia matrix
    Eigen::MatrixXd _Ac = Eigen::MatrixXd::Zero(_nx,_nx);     // Continuous state matrix
    Eigen::MatrixXd _Bc = Eigen::MatrixXd::Zero(_nx,_nu);     // Continuous input matrix
    Eigen::VectorXd _f = Eigen::VectorXd(_nx);                // Constant offset matrix
    Eigen::VectorXd _f_t = Eigen::VectorXd::Zero(_nx);        // Time-variant offset matrix
    Eigen::MatrixXd _A = Eigen::MatrixXd(_nx,_nx);            // Discrete state matrix
    Eigen::MatrixXd _B = Eigen::MatrixXd(_nx,_nu);            // Discrete input matrix
    Eigen::MatrixXd _C = Eigen::MatrixXd::Zero(_ny,_nu);      // Output matrix
    Eigen::MatrixXd _q = Eigen::MatrixXd(_ny,_ny);            // Output weighting matrix
    Eigen::MatrixXd _r = Eigen::MatrixXd(_ny,_ny);            // Control input weighting matrix
    Eigen::MatrixXd _I = Eigen::MatrixXd::Identity(3,3);      // 3x3 identity matrix
    Eigen::Vector3d _1_3 = Eigen::Vector3d(0,0,1);            // Z-axis vector
    al::real_1d_array _U;                                     // Control input prediction

    // Partial state matrices
    Eigen::MatrixXd _A_rv = _I;
    Eigen::MatrixXd _A_vp = Eigen::MatrixXd(3,3);
    Eigen::MatrixXd _A_pp = Eigen::MatrixXd(3,3);
    Eigen::MatrixXd _A_pw = _I;
    Eigen::MatrixXd _A_ww = Eigen::MatrixXd(3,3);

    // Partial input matrices
    Eigen::Vector3d _B_vt;
    Eigen::Matrix3d _B_wm;

    // Partial offset matrices
    Eigen::Vector3d _f_r;
    Eigen::Vector3d _f_v;
    Eigen::Vector3d _f_p;
    Eigen::Vector3d _f_p_t;
    Eigen::Vector3d _f_w;
    Eigen::Vector3d _gamma;

    // Constants
    double _m = 1.05;                                         // Mass (kg)
    double _g = 9.81;                                         // Gravity (m/s)
    double _j_xx = 0.0122;                                    // x-axis principle moment
    double _j_yy = 0.0126;                                    // y-axis principle moment
    double _j_zz = 0.0239;                                    // z-axis principle moment


  public:
    Eigen::VectorXd x = Eigen::VectorXd(18);                  // State vector
    Eigen::VectorXd y = Eigen::VectorXd(_ny);                 // Output vector
    Eigen::VectorXd u = Eigen::VectorXd(_nu);                 // Control input vector
    Eigen::VectorXd u_lin = Eigen::VectorXd(_nu);             // Linearization control input

  public:

    // Default constructor
    LieMPC()
    {
      // Set default parameter values
      _np = 32;
      _ts = 0.02;
      _J(0,0) = _j_xx;
      _J(1,1) = _j_yy;
      _J(2,2) = _j_zz;
      _J_inv = _J.inverse();
      _B_wm = _J_inv;

      // Set default cost weights
      Eigen::Matrix <double,6,1> y_weight;
      y_weight << 1, 1, 1, 0.1, 0.1, 0.1;
      _q = y_weight.asDiagonal();

      Eigen::Matrix <double,4,1> u_weight;
      u_weight << 1, 1, 1, 1;
      _r = u_weight.asDiagonal();

      // Set matrix sizes
      _U.setlength(_np*_nu);
    };

    // Constructor used to specify prediction horizon, timestep, and weights
    LieMPC(int n_p, double timestep, Eigen::MatrixXd y_weight, Eigen::MatrixXd u_weight)
    {
      // Fetch paramter values
      _np = n_p;
      _ts = timestep;
      _q = y_weight;
      _r = u_weight;
      _J(0,0) = _j_xx;
      _J(1,1) = _j_yy;
      _J(2,2) = _j_zz;
      _J_inv = _J.inverse();
      _B_wm = _J_inv;

      // Set matrix sizes
      _U.setlength(_np*_nu);
    };

    void linearize() {
      // Build rotation matrix
      Eigen::Matrix3d _Rotm = Eigen::Map<Eigen::Matrix3d>((x.segment(6,9)).data()).transpose();

      // Build state matrix components
      _A_vp = (1/_m)*_Rotm*_cross_op(_1_3*u_lin(0));
      _A_pp = -_cross_op(x.segment(15,3));
      _A_ww = _J_inv*(_cross_op(_J*x.segment(15,3))-(_cross_op(x.segment(15,3))*_J));

      // Build input matrix components
      _B_vt = -(1/_m)*_Rotm*_1_3;

      // Build offset matrix componenets
      _f_r = -x.segment(3,3);
      _f_v = (1/_m)*_Rotm*_1_3*u_lin(0)-_1_3*_g;
      _f_p = -x.segment(15,3);
      _gamma = (-_cross_op(x.segment(15,3))*_J*x.segment(15,3) + u_lin.segment(1,3));
      _f_p_t = -_J_inv*_gamma;
      _f_w = _f_p_t;
    }

  private:

    Eigen::Matrix3d _cross_op(Eigen::Vector3d _v) {
      Eigen::MatrixXd _M = Eigen::MatrixXd::Zero(3,3);
      _M(0,1) = -_v(2);
      _M(0,2) = _v(1);
      _M(1,2) = -_v(0);
      _M(1,0) = _v(2);
      _M(2,0) = -_v(1);
      _M(2,1) = -_v(0);
      return _M;
    }
};
