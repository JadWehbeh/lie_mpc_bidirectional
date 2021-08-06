# include <optimization.h>
# include <Eigen/Dense>
# include <unsupported/Eigen/MatrixFunctions>
# include <math.h>
# include <iostream>


namespace al = alglib;

class LieMPC {

  public:

    Eigen::Matrix<double,18,1> x;                             // State vector
    Eigen::Matrix<double,6,1> y;                              // Output vector
    Eigen::Vector4d u;                                        // Control input vector
    Eigen::Vector4d u_lin;                                    // Linearization control input

  private:

    int _nx = 12;                                             // Number of states
    int _ny = 6;                                              // Number of outputs
    int _nu = 4;                                              // Number of control inputs
    int _np;                                                  // Prediction steps
    double _ts;                                               // Time step (s)
    double _T_max;                                            // Maximum thrust (N)
    double _t_ramp;                                           // Thrust ramp time (s)
    Eigen::Matrix3d _J = Eigen::Matrix3d::Zero();             // Inertia matrix (kg.m^2)
    Eigen::Matrix3d _J_inv;                                   // Inverse of inertia matrix
    Eigen::Matrix<double,12,12> _Ac =
      Eigen::Matrix<double,12,12>::Zero();                    // Continuous state matrix
    Eigen::Matrix<double,12,4> _Bc =
      Eigen::Matrix<double,12,4>::Zero();                     // Continuous input matrix
    Eigen::Matrix<double,6,12> _Cc =
      Eigen::Matrix<double,6,12>::Zero();                     // Continuous output matrix
    Eigen::Matrix<double,12,1> _f;                            // Constant offset matrix
    Eigen::Matrix<double,12,1> _f_t;                          // Time-variant offset matrix
    Eigen::Matrix<double,12,12> _A;                           // Discrete state matrix
    Eigen::Matrix<double,12,4> _B;                            // Discrete input matrix
    Eigen::Matrix<double,6,12> _C;                            // Discrete output matrix
    Eigen::Matrix<double,6,4> _D;                             // Discrete feedthrough matrix
    Eigen::Matrix<double,6,6> _q;                             // Output weighting matrix
    Eigen::Matrix<double,4,4> _r;                             // Control input weighting matrix
    Eigen::Matrix3d _I = Eigen::Matrix3d::Identity();         // 3x3 identity matrix
    Eigen::Vector3d _1_3 = Eigen::Vector3d(0,0,1);            // Z-axis vector
    al::real_1d_array _U;                                     // Control input prediction

    // Partial state matrices
    Eigen::Matrix3d _A_rv = _I;
    Eigen::Matrix3d _A_vp;
    Eigen::Matrix3d _A_pp;
    Eigen::Matrix3d _A_pw = _I;
    Eigen::Matrix3d _A_ww;

    // Partial input matrices
    Eigen::Vector3d _B_vt;
    Eigen::Matrix3d _B_wm;

    // Partial offset matrices
    Eigen::Vector3d _f_r;
    Eigen::Vector3d _f_v;
    Eigen::Vector3d _f_p;
    Eigen::Vector3d _f_p_t;
    Eigen::Vector3d _f_w;

    // MPC Matrices
    Eigen::MatrixXd _H;
    Eigen::VectorXd _F;
    Eigen::MatrixXd _G;
    Eigen::MatrixXd _C_bar;

    // Constants
    double _m = 1.0;                                         // Mass (kg)
    double _g = 9.81;                                         // Gravity (m/s)
    double _j_xx = 0.0122;                                    // x-axis principle moment
    double _j_yy = 0.0126;                                    // y-axis principle moment
    double _j_zz = 0.0239;                                    // z-axis principle moment

    // Calculation Variables;
    Eigen::Vector3d _gamma;
    Eigen::Matrix<double,12,12> _Ak;
    Eigen::Matrix<double,12,4> _ABk;
    Eigen::Matrix<double,12,1> _Afk;

    public:

    // Default constructor
    LieMPC()
    {
      // Set default parameter values
      _np = 32;
      _ts = 0.01;
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

      // Assign constant state matrix parts
      _Ac.block(0,3,3,3) = _A_rv;
      _Ac.block(6,9,3,3) = _A_pw;
      _Bc.block(9,1,3,3) = _B_wm;

      // Assign output matrix parts;
      _Cc.block(0,0,3,3) = _I;
      _Cc.block(3,6,3,3) = _I;


      // Set matrix sizes
      _H.resize(_nx*_np,_nu*_np);
      _F.resize(_nx*_np);
      _G.resize(_nx*_np,_nx);
      _C_bar.resize(_ny*_np,_nx*_np);
      _H = Eigen::MatrixXd::Zero(_nx*_np,_nu*_np);
      _F = Eigen::VectorXd::Zero(_nx*_np);
      _G = Eigen::MatrixXd::Zero(_nx*_np,_nx);
      _C_bar = Eigen::MatrixXd::Zero(_ny*_np,_nx*_np);
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

      // Assign constant state matrix parts
      _Ac.block(0,3,3,3) = _A_rv;
      _Ac.block(6,9,3,3) = _A_pw;
      _Bc.block(9,1,3,3) = _B_wm;

      // Assign output matrix parts;
      _Cc.block(0,0,3,3) = _I;
      _Cc.block(3,6,3,3) = _I;

      // Set matrix sizes
      _H.resize(_nx*_np,_nu*_np);
      _F.resize(_nx*_np);
      _G.resize(_nx*_np,_nx);
      _C_bar.resize(_ny*_np,_nx*_np);
      _H = Eigen::MatrixXd::Zero(_nx*_np,_nu*_np);
      _F = Eigen::VectorXd::Zero(_nx*_np);
      _G = Eigen::MatrixXd::Zero(_nx*_np,_nx);
      _C_bar = Eigen::MatrixXd::Zero(_ny*_np,_nx*_np);
      _U.setlength(_np*_nu);
    };

    void linearize(){
      // Build rotation matrix
      Eigen::Matrix3d _Rotm = Eigen::Map<Eigen::Matrix3d>((x.segment(6,9)).data()).transpose();

      // Build state matrix components
      _A_vp.noalias() = (1./_m)*_Rotm*_cross_op(_1_3*u_lin(0));
      _A_pp.noalias() = -_cross_op(x.segment(15,3));
      _A_ww.noalias() = _J_inv*(_cross_op(_J*x.segment(15,3))-(_cross_op(x.segment(15,3))*_J));

      // Build input matrix components
      _B_vt.noalias() = -(1./_m)*_Rotm*_1_3;

      // Build offset matrix componenets
      _f_r = -x.segment(3,3);
      _f_v.noalias() = (1./_m)*_Rotm*_1_3*u_lin(0)-_1_3*_g;
      _f_p = -x.segment(15,3);
      _gamma.noalias() = (-_cross_op(x.segment(15,3))*_J*x.segment(15,3) + u_lin.segment(1,3));
      _f_p_t.noalias() = -_J_inv*_gamma;
      _f_w = _f_p_t;

      // Build matrices
      _Ac.block(3,6,3,3) = _A_vp;
      _Ac.block(6,6,3,3) = _A_pp;
      _Ac.block(9,9,3,3) = _A_ww;

      _Bc.block(3,0,3,1) = _B_vt;

      _f.segment(0,3) = _f_r;
      _f.segment(3,3) = _f_v;
      _f.segment(6,3) = _f_p;
      _f.segment(9,3) = _f_w;

      _f_t.segment(6,3) = _f_p_t;

//      std::cout << _Ac << std::endl;
//      std::cout << _Bc << std::endl;
//      std::cout << _f + _f_t*_ts << std::endl;
    };

    void discretize(){
      // Obtain discrete A and B matrices using zero order hold
//      _A.noalias() = (_Ac*_ts).exp();
//      _B.noalias() = ((_Ac.transpose()).householderQr().solve((_A -
//            Eigen::MatrixXd::Identity(_nx,_nx)).transpose())).transpose()*_Bc;

      _A.noalias() = (Eigen::Matrix<double,12,12>::Identity() -
          (_ts/2.)*_Ac).householderQr().solve(
            Eigen::Matrix<double,12,12>::Identity() + (_ts/2)*_Ac);
      _B.noalias() = 2.*(Eigen::Matrix<double,12,12>::Identity() -
          (_ts/2.)*_Ac).householderQr().solve((_ts/2.)*_Bc);
      _C.noalias() = 0.5*_Cc*(Eigen::Matrix<double,12,12>::Identity() + _A);
      _D.noalias() = 0.5*_Cc*_B;

      std::cout << _A << std::endl;
      std::cout << _B << std::endl;
      std::cout << _C << std::endl;
      std::cout << _D << std::endl;
    };

    void build_mpc(){

      _Ak = Eigen::MatrixXd::Identity(_nx,_nx);
      _ABk = _B;
      _Afk = _ts*(_f + (_ts/2)*_f_t);

      for (int i = 0; i<_np; i++) {

        if (i != 0) {
          _Ak *= _A;
          _ABk.noalias() = _Ak*_B;
          _Afk.noalias() = _Ak*_ts*(_f + i*(_ts/2)*_f_t);
        }

        _C_bar.block(i*_ny,i*_nx,_ny,_nx) = _C;
        _G.block(i*_nx,0,_nx,_nx) = _Ak;

        for (int j = 0; j + i < _np - 1; j++) {
          _F.segment(_nx*(_np-1-j),_nx) += _Afk;
          _H.block((j+i+1)*_nx,j*_nu,_nx,_nu) = _ABk;
        }
      }
    };

  private:

    Eigen::Matrix3d _cross_op(Eigen::Vector3d _v) {
      // Calculate cross operator matrix from vector
      Eigen::MatrixXd _M = Eigen::MatrixXd::Zero(3,3);
      _M(0,1) = -_v(2);
      _M(0,2) = _v(1);
      _M(1,2) = -_v(0);
      _M(1,0) = _v(2);
      _M(2,0) = -_v(1);
      _M(2,1) = -_v(0);
      return _M;
    };
};
