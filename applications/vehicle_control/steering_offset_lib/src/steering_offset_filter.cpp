#include "controller/steering_offset/steering_offset_filter.h"

namespace pnc {
namespace control {

void SteeringOffsetFilter::Init(const SteeringOffsetConfig& config) {
  config_ = config;
  Reset();
}

void SteeringOffsetFilter::Reset() {
  x_ = 0.0;
  P_ = 1e-4;
  last_wheel_angle_ = 0.0;
  last_gate_pass_ = false;
  output_enabled_ = false;
  valid_duration_ = 0.0;
}

bool SteeringOffsetFilter::CheckUpdateGate(double vel_ego, double wheel_angle,
                                           double yawrate) const {
  if (vel_ego < config_.velocity_threshold) {
    return false;
  }
  if (std::abs(wheel_angle) > config_.max_wheel_angle) {
    return false;
  }
  if (std::abs(yawrate) > config_.max_yawrate) {
    return false;
  }
  if (std::abs(vel_ego * yawrate) > config_.max_lateral_accel) {
    return false;
  }
  double wheel_angle_rate =
      std::abs(wheel_angle - last_wheel_angle_) / config_.dt;
  if (wheel_angle_rate > config_.max_wheel_angle_rate) {
    return false;
  }
  return true;
}

void SteeringOffsetFilter::Update(double vel_ego, double wheel_angle,
                                  double yawrate) {
  // STEP 1: check update gate
  bool gate_pass = CheckUpdateGate(vel_ego, wheel_angle, yawrate);
  last_wheel_angle_ = wheel_angle;
  last_gate_pass_ = gate_pass;

  if (!gate_pass) {
    // Gate not satisfied: freeze KF, but keep tracking valid duration
    if (is_converged()) {
      valid_duration_ += config_.dt;
    }
    if (valid_duration_ >= config_.min_valid_duration) {
      output_enabled_ = true;
    }
    return;
  }

  const double L = config_.wheelbase;
  const double Q = config_.offset_model_error_variance;
  const double R = config_.yawrate_measure_variance;

  // STEP 2: Predict
  double x_prior = x_;
  double P_prior = P_ + Q;

  // STEP 3: Update
  double H = -vel_ego / L;
  double r_predicted = vel_ego / L * (wheel_angle - x_prior);
  double y = yawrate - r_predicted;
  double S = H * H * P_prior + R;
  double K = P_prior * H / S;

  x_ = x_prior + K * y;
  P_ = (1.0 - K * H) * P_prior;

  // Track convergence duration
  if (is_converged()) {
    valid_duration_ += config_.dt;
  } else {
    valid_duration_ = 0.0;
  }

  if (valid_duration_ >= config_.min_valid_duration) {
    output_enabled_ = true;
  }
}

SteeringOffsetFilter::CompensationResult
SteeringOffsetFilter::ComputeCompensation(double steering_angle_cmd,
                                          double steer_ratio) const {
  CompensationResult result{};

  if (is_valid()) {
    result.steering_angle_bias = -x_ * steer_ratio;
    result.steering_angle_dist =
        std::clamp(result.steering_angle_bias, -config_.bias_limit,
                   config_.bias_limit);
    result.steering_cmd_corrected =
        steering_angle_cmd - result.steering_angle_dist;
    result.wheel_angle_bias = result.steering_angle_dist / steer_ratio;
  } else {
    result.steering_angle_bias = 0.0;
    result.steering_angle_dist = 0.0;
    result.steering_cmd_corrected = steering_angle_cmd;
    result.wheel_angle_bias = 0.0;
  }

  return result;
}

double SteeringOffsetFilter::CorrectedWheelAngle(double wheel_angle,
                                                  double steer_ratio) const {
  if (is_valid()) {
    double wheel_angle_limit = config_.bias_limit / steer_ratio;
    double offset_limited =
        std::clamp(x_, -wheel_angle_limit, wheel_angle_limit);
    return wheel_angle - offset_limited;
  }
  return wheel_angle;
}

}  // namespace control
}  // namespace pnc
