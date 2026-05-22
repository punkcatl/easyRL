#include "controller/steering_offset/steering_offset_filter.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

int main() {
  using namespace pnc::control;

  constexpr double L = 2.9;
  constexpr double SR = 13.5;
  constexpr double dt = 0.02;

  // Injected bias: 10 deg steering wheel -> wheel angle level
  constexpr double steering_bias_deg = 10.0;
  constexpr double wheel_angle_bias_true = steering_bias_deg / 57.3 / SR;

  std::mt19937 rng(42);
  std::normal_distribution<double> yawrate_noise(0.0, 0.001);

  SteeringOffsetConfig config;
  config.wheelbase = L;
  config.dt = dt;

  SteeringOffsetFilter filter;
  filter.Init(config);

  constexpr double total_time = 60.0;
  constexpr int total_steps = static_cast<int>(total_time / dt);

  std::ofstream csv("sim_output.csv");
  csv << "time,vel_ego,wheel_angle_sensor,wheel_angle_true,yawrate_true,"
         "yawrate_noisy,x_estimate,P,gate_pass,output_enabled,"
         "valid_duration,wheel_angle_rate\n";

  double last_wheel_angle_sensor = 0.0;

  for (int i = 0; i < total_steps; ++i) {
    double t = i * dt;

    // --- Velocity profile ---
    double vel_ego = 0.0;
    if (t < 5.0) {
      vel_ego = 20.0 * t / 5.0;
    } else if (t >= 55.0) {
      vel_ego = 20.0 - 17.0 * (t - 55.0) / 5.0;
    } else {
      vel_ego = 20.0;
    }

    // --- True wheel angle (driver intent) ---
    double wheel_angle_true = 0.0;
    if (t >= 30.0 && t < 35.0) {
      double phase = (t - 30.0) / 5.0 * 2.0 * M_PI;
      wheel_angle_true = 0.03 * std::sin(phase);
    }

    // --- Sensor reading = true + bias ---
    double wheel_angle_sensor = wheel_angle_true + wheel_angle_bias_true;

    // --- True yawrate from kinematics ---
    double yawrate_true = (vel_ego > 0.1) ? vel_ego / L * wheel_angle_true : 0.0;

    // --- Noisy yawrate measurement ---
    double yawrate_noisy = yawrate_true + yawrate_noise(rng);

    // --- Wheel angle rate (for logging) ---
    double wheel_angle_rate = std::abs(wheel_angle_sensor - last_wheel_angle_sensor) / dt;
    last_wheel_angle_sensor = wheel_angle_sensor;

    // --- Run filter ---
    filter.Update(vel_ego, wheel_angle_sensor, yawrate_noisy);

    // --- Write CSV ---
    csv << t << ","
        << vel_ego << ","
        << wheel_angle_sensor << ","
        << wheel_angle_true << ","
        << yawrate_true << ","
        << yawrate_noisy << ","
        << filter.offset() << ","
        << filter.variance() << ","
        << (filter.last_gate_pass() ? 1 : 0) << ","
        << (filter.is_valid() ? 1 : 0) << ","
        << filter.valid_duration() << ","
        << wheel_angle_rate
        << "\n";
  }

  csv.close();
  std::cout << "Simulation complete. Output: sim_output.csv\n";
  std::cout << "True wheel angle bias: " << wheel_angle_bias_true << " rad ("
            << steering_bias_deg << " deg steering wheel)\n";
  std::cout << "Final estimate: " << filter.offset() << " rad\n";
  std::cout << "Final P: " << filter.variance() << "\n";
  std::cout << "Output enabled: " << (filter.is_valid() ? "yes" : "no") << "\n";

  return 0;
}
