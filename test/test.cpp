#include <lie_mpc_bidirectional.hpp>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE("object_creation")
{
  LieMPC test_controller;
  REQUIRE(test_controller.nx == 12);
}
