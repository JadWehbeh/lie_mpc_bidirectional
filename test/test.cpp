#include <Eigen/Dense>
#include <lie_mpc_bidirectional.hpp>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <chrono>
#include <iostream>

TEST_CASE("Hover")
{
  Eigen::VectorXd x_lin = Eigen::VectorXd(18);
  Eigen::VectorXd u_lin = Eigen::VectorXd(4);
  Eigen::VectorXd Y = Eigen::VectorXd::Zero(8 * 12);
  x_lin << 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.;
  u_lin << 2.4525, 2.4525, 2.4525, 2.4525;
  for (int i = 0; i < 8; i++) {
    Y(i * 12 + 3) = 1.;
    Y(i * 12 + 7) = 1.;
    Y(i * 12 + 11) = 1.;
  }
  LieMPC test_controller;
  test_controller.x = x_lin;
  test_controller.u_lin = u_lin;
  test_controller.Y = Y;
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  test_controller.linearize();
  test_controller.discretize();
  test_controller.build_mpc();
  test_controller.solve();
  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;
  REQUIRE((std::abs(test_controller.u(0) - 2.4525) < 0.05 && std::abs(test_controller.u(1) - 2.4525) < 0.05 && std::abs(test_controller.u(2) - 2.4525) < 0.05 && std::abs(test_controller.u(3) - 2.4525) < 0.05));
};

TEST_CASE("Inverted Hover")
{
  Eigen::VectorXd x_lin = Eigen::VectorXd(18);
  Eigen::VectorXd u_lin = Eigen::VectorXd(4);
  Eigen::VectorXd Y = Eigen::VectorXd::Zero(8 * 12);
  x_lin << 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., -1., 0., 0., 0., -1., 0., 0., 0.;
  u_lin << -2.4525, -2.4525, -2.4525, -2.4525;
  for (int i = 0; i < 8; i++) {
    Y(i * 12 + 3) = 1.;
    Y(i * 12 + 7) = -1.;
    Y(i * 12 + 11) = -1.;
  }
  LieMPC test_controller;
  test_controller.x = x_lin;
  test_controller.u_lin = u_lin;
  test_controller.Y = Y;
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  test_controller.linearize();
  test_controller.discretize();
  test_controller.build_mpc();
  test_controller.solve();
  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;
  REQUIRE((std::abs(test_controller.u(0) + 2.4525) < 0.05 && std::abs(test_controller.u(1) + 2.4525) < 0.05 && std::abs(test_controller.u(2) + 2.4525) < 0.05 && std::abs(test_controller.u(3) + 2.4525) < 0.05));
};

TEST_CASE("Consistency")
{
  Eigen::VectorXd x_lin = Eigen::VectorXd(18);
  Eigen::VectorXd u_lin = Eigen::VectorXd(4);
  Eigen::VectorXd u_init = Eigen::VectorXd(4);
  Eigen::VectorXd Y = Eigen::VectorXd::Zero(8 * 12);
  x_lin << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7071, -0.7071, 0., 0.6830, 0.6830, -0.2588, 0.1830, 0.1830, 0.9659, 0.1, 0.1, 0.1;
  u_lin << 2.4525, 2.4525, 2.4525, 2.4525;
  for (int i = 0; i < 8; i++) {
    Y(i * 12 + 3) = 1.;
    Y(i * 12 + 7) = 1.;
    Y(i * 12 + 11) = 1.;
  }
  LieMPC test_controller;
  test_controller.x = x_lin;
  test_controller.u_lin = u_lin;
  test_controller.Y = Y;
  std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
  test_controller.linearize();
  test_controller.discretize();
  test_controller.build_mpc();
  test_controller.solve();
  u_init = test_controller.u;
  for (int i = 0; i < 10; i++) {
    test_controller.linearize();
    test_controller.discretize();
    test_controller.build_mpc();
    test_controller.solve();
  }
  std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
  std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() << "[µs]" << std::endl;
  REQUIRE((std::abs(test_controller.u(0) - u_init(0)) < 1e-8 && std::abs(test_controller.u(1) - u_init(1)) < 1e-8 && std::abs(test_controller.u(2) - u_init(2)) < 1e-8 && std::abs(test_controller.u(3) - u_init(3)) < 1e-8));
};
