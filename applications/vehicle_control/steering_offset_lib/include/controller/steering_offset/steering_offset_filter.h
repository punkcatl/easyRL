#pragma once

#include <cmath>
#include <algorithm>

namespace pnc {
namespace control {

struct SteeringOffsetConfig {
  // Vehicle
  double wheelbase = 2.9;

  // KF noise
  double offset_model_error_variance = 5e-11;
  double yawrate_measure_variance = 2e-6;

  // Validity
  double var_valid_threshold = 2e-8;
  // Minimum duration that the estimate has existed in a converged state (P < threshold)
  // before compensation is enabled. Freeze periods count toward this duration because
  // the estimate remains stable (unchanged). Once output_enabled_ is set, only Reset() clears it.
  double min_valid_duration = 2.0;

  // Update gate conditions
  double velocity_threshold = 5.0;         // m/s
  double max_wheel_angle = 0.1;            // rad (~5.7°)
  double max_yawrate = 0.1;               // rad/s (~5.7°/s)
  double max_lateral_accel = 1.0;          // m/s²
  double max_wheel_angle_rate = 0.1;       // rad/s

  // Compensation
  double bias_limit = 12.0 / 57.3;        // rad (~12°, steering wheel level)

  // Control period
  double dt = 0.02;                        // s (50 Hz)
};

class SteeringOffsetFilter {
 public:
  void Init(const SteeringOffsetConfig& config);
  void Reset();

  // STEP 1-3: KF update (called in preprocess, every cycle)
  void Update(double vel_ego, double wheel_angle, double yawrate);

  // STEP 4-6: compute compensation (called in control loop)
  struct CompensationResult {
    double steering_angle_bias;
    double steering_angle_dist;
    double wheel_angle_bias;
    double steering_cmd_corrected;
  };
  CompensationResult ComputeCompensation(double steering_angle_cmd,
                                         double steer_ratio) const;

  // Reset scenario: corrected wheel angle for MPC init state
  double CorrectedWheelAngle(double wheel_angle, double steer_ratio) const;

  // Accessors
  double offset() const { return x_; }
  double variance() const { return P_; }
  bool is_converged() const { return P_ < config_.var_valid_threshold; }
  bool is_valid() const { return output_enabled_; }
  bool last_gate_pass() const { return last_gate_pass_; }
  double valid_duration() const { return valid_duration_; }

 private:
  bool CheckUpdateGate(double vel_ego, double wheel_angle,
                       double yawrate) const;

  SteeringOffsetConfig config_;

  // KF state
  double x_ = 0.0;
  double P_ = 1e-4;

  // Update gate state
  double last_wheel_angle_ = 0.0;
  bool last_gate_pass_ = false;

  // Output enable logic
  bool output_enabled_ = false;
  double valid_duration_ = 0.0;
};

}  // namespace control
}  // namespace pnc
