# include "optimization.h"
# include <math.h>

namespace al = alglib;

class LieMPC {
  public:

    int nx;
    int ny;
    int nu;
    int np;
    double ts;
    al::real_1d_array x, y, u;
    al::real_2d_array Q, R;

    // Default constructor
    LieMPC()
    {
      n_x = 12;
      ny = 6;
      nu = 4;
      x.setlength(nx);
      y.setlength(ny);
      u.setlength(nu);
      Q.setlength(ny, ny);
      R.setlength(nu, nu);
      np = 32;
      ts = 0.02;
    }

    // Constructor used to specify prediction horizon, timestep, and weights
    LieMPC(int n_p, double timestep, al::real_2d_array y_weight, al::real_2d_array u_weight)
    {
      nx = 12;
      ny = 6;
      nu = 4;
      x.setlength(nx);
      y.setlength(ny);
      u.setlength(nu);
      np = n_p;
      ts = timestep;
      Q = y_weight;
      R = u_weight;
    }
}
