#include <hdl_localization/pose_estimator.hpp>

#include <pcl/filters/voxel_grid.h>
#include <hdl_localization/pose_system.hpp>
#include <hdl_localization/odom_system.hpp>
#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief constructor
 * @param registration        registration method
 * @param stamp               timestamp
 * @param pos                 initial position
 * @param quat                initial orientation
 * @param cool_time_duration  during "cool time", prediction is not performed
 */
PoseEstimator::PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const ros::Time& stamp, const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, MeasurementNoise measurement_noise_param, ProcessNoise process_noise_param, double cool_time_duration)
    : init_stamp(stamp), registration(registration), cool_time_duration(cool_time_duration), force_2d(true), robot_constant_z(0.00f) {

  last_observation = Eigen::Matrix4f::Identity();
  last_observation.block<3, 3>(0, 0) = quat.toRotationMatrix();
  last_observation.block<3, 1>(0, 3) = pos;

  process_noise = Eigen::MatrixXf::Identity(16, 16);
  process_noise.middleRows(0, 3) *= process_noise_param.pxyz;
  process_noise.middleRows(3, 3) *= process_noise_param.vxyz;
  process_noise.middleRows(6, 4) *= process_noise_param.qwxyz;
  process_noise.middleRows(10, 3) *= process_noise_param.acc_xyz;
  process_noise.middleRows(13, 3) *= process_noise_param.gyro_xyz;

  Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
  measurement_noise.middleRows(0, 3) *= measurement_noise_param.xyz;
  measurement_noise.middleRows(3, 4) *= measurement_noise_param.qwxyz;

  Eigen::VectorXf mean(16);
  mean.middleRows(0, 3) = pos;
  mean.middleRows(3, 3).setZero();
  mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());
  mean.middleRows(10, 3).setZero();
  mean.middleRows(13, 3).setZero();

  Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

  if(force_2d) {
    force_two_d_process_noise(process_noise);
    force_two_d_measurement_noise(measurement_noise);
    force_two_d_cov(cov);
    force_two_d_mean(mean);
    force_two_d_last_observation(last_observation);
  }

  PoseSystem system;
  ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>(system, 16, 6, 7, process_noise, measurement_noise, mean, cov));
}

void PoseEstimator::force_two_d_process_noise(Eigen::MatrixXf& entity) {
    entity(2, 2) = 0.000001; // position z
    entity(5, 5) = 0.000001; // velocity z
    entity(7, 7) = 0.000001; // roll
    entity(8, 8) = 0.000001; // pitch
    entity(12, 12) = 0.000001; // acceleration z
    entity(13, 13) = 0.000001; // angular velocity roll
    entity(14, 14) = 0.000001; // angular velocity pitch
}

void PoseEstimator::force_two_d_measurement_noise(Eigen::MatrixXf& entity){
    entity(2) = 0.000001; // position z
    entity(4) = 0.000001; // roll
    entity(5) = 0.000001; // pitch
}

void PoseEstimator::force_two_d_cov(Eigen::MatrixXf& entity){
    entity(2, 2) = 0.000001; // position z
    entity(5, 5) = 0.000001; // velocity z
    entity(7, 7) = 0.000001; // roll
    entity(8, 8) = 0.000001; // pitch
    entity(12, 12) = 0.000001; // acceleration z
    entity(13, 13) = 0.000001; // angular velocity roll
    entity(14, 14) = 0.000001; // angular velocity pitch
}

void PoseEstimator::force_two_d_mean(Eigen::VectorXf& entity) {
    entity(5) = 0.0f; // velocity z
    entity(7) = 0.0f; // roll
    entity(8) = 0.0f; // pitch
    entity(12) = 0.0f; // acceleration z
    entity(13) = 0.0f; // angular velocity roll
    entity(14) = 0.0f; // angular velocity pitch
}

void PoseEstimator::force_two_d_predict_odom(Eigen::VectorXf& entity) {
  entity(2) = 0.0f; // delta position z
  entity(4) = 0.0f; // delta roll
  entity(5) = 0.0f; // delta pitch
}

void PoseEstimator::force_two_d_imu(Eigen::VectorXf& entity) {
  entity(2) = 9.80665f; // acceleration z
  entity(3) = 0.0f; // velocity roll
  entity(4) = 0.0f; // velocity pitch
}

void PoseEstimator::force_two_d_last_observation(Eigen::Matrix4f& entity) {
  Eigen::Vector3f euler = entity.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
  Eigen::Matrix3f yaw_rot_mat;
  yaw_rot_mat = Eigen::AngleAxisf(euler.z(), Eigen::Vector3f::UnitZ());
  entity.block<3, 3>(0, 0) = yaw_rot_mat;
  entity(3, 3) = robot_constant_z;  // position z
}

PoseEstimator::~PoseEstimator() {}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const ros::Time& stamp) {
  if ((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
    prev_stamp = stamp;
    return;
  }

  double dt = (stamp - prev_stamp).toSec();
  prev_stamp = stamp;

  ukf->setProcessNoiseCov(process_noise * dt);
  ukf->system.dt = dt;
  ukf->predict();
}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const ros::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
  if ((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
    prev_stamp = stamp;
    return;
  }

  double dt = (stamp - prev_stamp).toSec();
  prev_stamp = stamp;

  Eigen::VectorXf control(6);
  control.head<3>() = acc;
  control.tail<3>() = gyro;

  if(force_2d){
    force_two_d_imu(control);
  }

  ukf->setProcessNoiseCov(process_noise * dt);
  ukf->system.dt = dt;
  ukf->predict(control);
}

/**
 * @brief update the state of the odomety-based pose estimation
 */
void PoseEstimator::predict_odom(const Eigen::Matrix4f& odom_delta) {
  if(!odom_ukf) {
    Eigen::MatrixXf odom_process_noise = Eigen::MatrixXf::Identity(7, 7);
    Eigen::MatrixXf odom_measurement_noise = Eigen::MatrixXf::Identity(7, 7) * 1e-3;

    Eigen::VectorXf odom_mean(7);
    odom_mean.block<3, 1>(0, 0) = Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
    odom_mean.block<4, 1>(3, 0) = Eigen::Vector4f(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]);
    Eigen::MatrixXf odom_cov = Eigen::MatrixXf::Identity(7, 7) * 1e-2;

    OdomSystem odom_system;
    odom_ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, OdomSystem>(odom_system, 7, 7, 7, odom_process_noise, odom_measurement_noise, odom_mean, odom_cov));
  }

  // invert quaternion if the rotation axis is flipped
  Eigen::Quaternionf quat(odom_delta.block<3, 3>(0, 0));
  if(odom_quat().coeffs().dot(quat.coeffs()) < 0.0) {
    quat.coeffs() *= -1.0f;
  }

  Eigen::VectorXf control(7);
  control.middleRows(0, 3) = odom_delta.block<3, 1>(0, 3);
  control.middleRows(3, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());

  Eigen::MatrixXf process_noise = Eigen::MatrixXf::Identity(7, 7);
  process_noise.topLeftCorner(3, 3) = Eigen::Matrix3f::Identity() * odom_delta.block<3, 1>(0, 3).norm() + Eigen::Matrix3f::Identity() * 1e-3;
  process_noise.bottomRightCorner(4, 4) = Eigen::Matrix4f::Identity() * (1 - std::abs(quat.w())) + Eigen::Matrix4f::Identity() * 1e-3;

  if(force_2d){
    force_two_d_predict_odom(control);
  }

  odom_ukf->setProcessNoiseCov(process_noise);
  odom_ukf->predict(control);
}

/**
 * @brief correct
 * @param cloud   input cloud
 * @return cloud aligned to the globalmap
 */
pcl::PointCloud<PoseEstimator::PointT>::Ptr PoseEstimator::correct(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
  last_correction_stamp = stamp;

  Eigen::Matrix4f no_guess = last_observation;
  Eigen::Matrix4f imu_guess;
  Eigen::Matrix4f odom_guess;
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

  if(!odom_ukf) {
    init_guess = imu_guess = matrix();
  } else {
    imu_guess = matrix();
    odom_guess = odom_matrix();

    Eigen::VectorXf imu_mean(7);
    Eigen::MatrixXf imu_cov = Eigen::MatrixXf::Identity(7, 7);
    imu_mean.block<3, 1>(0, 0) = ukf->mean.block<3, 1>(0, 0);
    imu_mean.block<4, 1>(3, 0) = ukf->mean.block<4, 1>(6, 0);

    imu_cov.block<3, 3>(0, 0) = ukf->cov.block<3, 3>(0, 0);
    imu_cov.block<3, 4>(0, 3) = ukf->cov.block<3, 4>(0, 6);
    imu_cov.block<4, 3>(3, 0) = ukf->cov.block<4, 3>(6, 0);
    imu_cov.block<4, 4>(3, 3) = ukf->cov.block<4, 4>(6, 6);

    Eigen::VectorXf odom_mean = odom_ukf->mean;
    Eigen::MatrixXf odom_cov = odom_ukf->cov;

    if (imu_mean.tail<4>().dot(odom_mean.tail<4>()) < 0.0) {
      odom_mean.tail<4>() *= -1.0;
    }

    Eigen::MatrixXf inv_imu_cov = imu_cov.inverse();
    Eigen::MatrixXf inv_odom_cov = odom_cov.inverse();

    Eigen::MatrixXf fused_cov = (inv_imu_cov + inv_odom_cov).inverse();
    Eigen::VectorXf fused_mean = fused_cov * inv_imu_cov * imu_mean + fused_cov * inv_odom_cov * odom_mean;

    init_guess.block<3, 1>(0, 3) = Eigen::Vector3f(fused_mean[0], fused_mean[1], fused_mean[2]);
    init_guess.block<3, 3>(0, 0) = Eigen::Quaternionf(fused_mean[3], fused_mean[4], fused_mean[5], fused_mean[6]).normalized().toRotationMatrix();
  }

  pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
  registration->setInputSource(cloud);
  registration->align(*aligned, init_guess);

  Eigen::Matrix4f trans = registration->getFinalTransformation();
  Eigen::Vector3f p = trans.block<3, 1>(0, 3);
  Eigen::Quaternionf q(trans.block<3, 3>(0, 0));

  if(quat().coeffs().dot(q.coeffs()) < 0.0f) {
    q.coeffs() *= -1.0f;
  }

  Eigen::VectorXf observation(7);
  observation.middleRows(0, 3) = p;
  observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());
  last_observation = trans;

  Eigen::Matrix4f no_guess_trans = registration->getFinalTransformation();
  wo_pred_error = no_guess.inverse() * no_guess_trans;
  ukf->correct(observation);

  Eigen::Matrix4f imu_guess_trans = registration->getFinalTransformation();
  imu_pred_error = imu_guess.inverse() * imu_guess_trans;

  if(odom_ukf) {
    if (observation.tail<4>().dot(odom_ukf->mean.tail<4>()) < 0.0) {
      odom_ukf->mean.tail<4>() *= -1.0;
    }

    odom_ukf->correct(observation);
    ukf->correct(observation);

    Eigen::Matrix4f odom_guess_trans = registration->getFinalTransformation();
    odom_pred_error = odom_guess.inverse() * odom_guess_trans;
  }

  return aligned;
}

/* getters */
ros::Time PoseEstimator::last_correction_time() const {
  return last_correction_stamp;
}

Eigen::Vector3f PoseEstimator::pos() const {
  if (force_2d) {
    return Eigen::Vector3f(ukf->mean[0], ukf->mean[1], robot_constant_z);
  } else {
    return Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
  }
}

Eigen::Vector3f PoseEstimator::vel() const {
  if (force_2d) {
    return Eigen::Vector3f(ukf->mean[3], ukf->mean[4], 0.0f);
  } else {
    return Eigen::Vector3f(ukf->mean[3], ukf->mean[4], ukf->mean[5]);
  }
}

Eigen::Quaternionf PoseEstimator::quat() const {
  if (force_2d) {
    return Eigen::Quaternionf(ukf->mean[6], 0.0f, 0.0f, ukf->mean[9]).normalized();
  } else {
    return Eigen::Quaternionf(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]).normalized();
  }
}

Eigen::Matrix4f PoseEstimator::matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = pos();
  return m;
}

Eigen::Vector3f PoseEstimator::odom_pos() const {
  if (force_2d) {
    return Eigen::Vector3f(odom_ukf->mean[0], odom_ukf->mean[1], 0.0f);
  } else {
    return Eigen::Vector3f(odom_ukf->mean[0], odom_ukf->mean[1], odom_ukf->mean[2]);
  }
}

Eigen::Quaternionf PoseEstimator::odom_quat() const {
  if (force_2d) {
    return Eigen::Quaternionf(odom_ukf->mean[3], 0.0f, 0.0f, odom_ukf->mean[6]).normalized();
  } else {
    return Eigen::Quaternionf(odom_ukf->mean[3], odom_ukf->mean[4], odom_ukf->mean[5], odom_ukf->mean[6]).normalized();
  }
}

Eigen::Matrix4f PoseEstimator::odom_matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = odom_quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = odom_pos();
  return m;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::wo_prediction_error() const {
  return wo_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::imu_prediction_error() const {
  return imu_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::odom_prediction_error() const {
  return odom_pred_error;
}
}