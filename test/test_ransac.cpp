/*
 * Copyright (c) 2015, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: Erik Nelson            ( eanelson@eecs.berkeley.edu )
 *          David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

#include <camera/camera.h>
#include <camera/camera_extrinsics.h>
#include <camera/camera_intrinsics.h>
#include <geometry/eight_point_algorithm_solver.h>
#include <matching/feature_match.h>
#include <matching/pairwise_image_match.h>
#include <math/random_generator.h>
#include <ransac/fundamental_matrix_ransac_problem.h>
#include <ransac/ransac.h>
#include <ransac/ransac_options.h>

#include <gtest/gtest.h>

namespace bsfm {

class TestRansac : public ::testing::Test {
 protected:

  // Make two camera views, and then create features and match them between the
  // views.
  PairwiseImageMatch CreateFakeMatchedImagePair(int num_features) {
    // Create a random number generator.
    math::RandomGenerator rng(math::RandomGenerator::Seed());

    // Create two cameras. By default they will have identity extrinsics.
    Camera camera1;
    Camera camera2;

    // Give the two cameras the same intrinsics.
    CameraIntrinsics intrinsics;
    intrinsics.SetImageLeft(0);
    intrinsics.SetImageTop(0);
    intrinsics.SetImageWidth(kImageWidth_);
    intrinsics.SetImageHeight(kImageHeight_);
    intrinsics.SetVerticalFOV(kVerticalFov_);
    intrinsics.SetFU(intrinsics.f_v());
    intrinsics.SetCU(0.5 * kImageWidth_);
    intrinsics.SetCV(0.5 * kImageHeight_);

    camera1.SetIntrinsics(intrinsics);
    camera2.SetIntrinsics(intrinsics);

    // Translate the 2nd camera along its X-axis to give it a known baseline.
    // Camera 2 will now be 2.0 units to the right of camera 1.
    camera2.MutableExtrinsics().TranslateX(2.0);

    // Create a bunch of points in 3D.
    PairwiseImageMatch matched_images_out;
    while(matched_images_out.feature_matches_.size() < num_features) {
      // Since the camera's +Z faces down the world's -Y direction, make random
      // points back there somewhere.
      double x_world = rng.DoubleUniform(-5.0, 7.0);
      double y_world = rng.DoubleUniform(-30.0, -20.0);
      double z_world = rng.DoubleUniform(-5.0, 5.0);

      // Project each of the 3D points into the two cameras.
      double u1 = 0.0, v1 = 0.0;
      double u2 = 0.0, v2 = 0.0;
      bool in_camera1 = camera1.WorldToImage(x_world, y_world, z_world, &u1, &v1);
      bool in_camera2 = camera2.WorldToImage(x_world, y_world, z_world, &u2, &v2);

      // Make sure that the point is visible to both cameras. We want feature
      // matches.
      if (!(in_camera1 && in_camera2)) {
        continue;
      }

      // Store this as a feature match.
      FeatureMatch match;
      match.feature1_.u_ = u1;
      match.feature1_.v_ = v1;
      match.feature2_.u_ = u2;
      match.feature2_.v_ = v2;
      matched_images_out.feature_matches_.push_back(match);
    }

    return matched_images_out;
  }

 private:
  const int kImageWidth_ = 1920;
  const int kImageHeight_ = 1080;
  const double kVerticalFov_ = 90.0 * M_PI / 180.0;
};  //\class TestNaiveFeatureMatcher


TEST_F(TestRansac, TestFundamentalMatrixNoiseless) {

  // Get some data. As input, we want a set of features that are matched across
  // two images.
  const int kNumFeatures = 8;
  PairwiseImageMatch data = CreateFakeMatchedImagePair(kNumFeatures);

  // Define the RANSAC problem - we are attempting to determine the fundamental
  // matrix for a set of noiseless feature correspondences in images.
  FundamentalMatrixRansacProblem problem;

  // Store data in the RANSAC problem format.
  std::vector<RansacDataElement<FeatureMatch> > ransac_data;
  for (const auto& feature_match : data.feature_matches_) {
    RansacDataElement<FeatureMatch> ransac_data_element(feature_match);
    ransac_data.push_back(ransac_data_element);
  }
  problem.SetData(ransac_data);

  // Create the ransac solver, set options, and run RANSAC on the problem.
  Ransac<FeatureMatch, FundamentalMatrixRansacModel> solver;
  RansacOptions options;

  // It should only take one iteration to find the right model.
  // Every feature match should be considered an inlier with such a huge error.
  options.iterations = 1;
  options.acceptable_error = 100.0;
  options.minimum_num_inliers = 8;

  solver.SetOptions(options);
  solver.Run(problem);

  // Get the solution from the problem object.
  ASSERT_TRUE(problem.SolutionFound());

  // If check not needed, but useful to show how to implement this.
  Eigen::Matrix3d fundamental_matrix_ransac(Eigen::Matrix3d::Identity());
  if (problem.SolutionFound()) {
    const FundamentalMatrixRansacModel& model =
        dynamic_cast<const FundamentalMatrixRansacModel&>(problem.Model());
    fundamental_matrix_ransac = model.F_;
  }

  // There was no noise in the matches, so this should be the same thing we
  // would have gotten if we were to run the fundamental matrix solver without
  // RANSAC.
  EightPointAlgorithmSolver solver2;
  FundamentalMatrixSolverOptions options2;
  solver2.SetOptions(options2);

  Eigen::Matrix3d fundamental_matrix_ep(Eigen::Matrix3d::Identity());
  ASSERT_TRUE(solver2.ComputeFundamentalMatrix(data.feature_matches_,
                                               fundamental_matrix_ep));

  // Make sure RANSAC comes up with the same solution.
  EXPECT_TRUE(fundamental_matrix_ep.isApprox(fundamental_matrix_ransac, 1e-8));
}

TEST_F(TestRansac, TestNeedAtLeastEight) {
  // Make sure that the fundamental matrix ransac solver fails when we don't
  // have a sufficient number of input matches. Make sure the solver fails
  // if we have >= 8 noiseless input matches.
  for (int ii = 0; ii <= 9; ++ii) {
    PairwiseImageMatch data = CreateFakeMatchedImagePair(ii);

    // Define the RANSAC problem.
    FundamentalMatrixRansacProblem problem;

    // Store data in the RANSAC problem format.
    std::vector<RansacDataElement<FeatureMatch> > ransac_data;
    for (const auto& feature_match : data.feature_matches_) {
      RansacDataElement<FeatureMatch> ransac_data_element(feature_match);
      ransac_data.push_back(ransac_data_element);
    }
    problem.SetData(ransac_data);

    // Create the ransac solver, set options, and run RANSAC on the problem.
    Ransac<FeatureMatch, FundamentalMatrixRansacModel> solver;
    RansacOptions options;

    // The 8 minimum inlier option is what will cause RANSAC to fail with less
    // than 8 matches.
    options.iterations = 10;
    options.acceptable_error = 100.0;
    options.minimum_num_inliers = 8;

    solver.SetOptions(options);
    solver.Run(problem);

    // Make sure we don't get a solution until we have at least 8 matches.
    if (ii < 8) {
      ASSERT_FALSE(problem.SolutionFound());
    } else {
      ASSERT_TRUE(problem.SolutionFound());
    }
  }
}

}  //\namespace bsfm
