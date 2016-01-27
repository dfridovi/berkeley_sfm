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
#include <matching/feature_matcher_options.h>
#include <matching/fast_matcher_2d3d.h>
#include <geometry/point_3d.h>
#include <geometry/rotation.h>
#include <matching/feature.h>
#include <math/random_generator.h>
#include <sfm/view.h>
#include <slam/landmark.h>
#include <slam/observation.h>

#include <gtest/gtest.h>

namespace bsfm {

  using Eigen::Matrix3d;
  using Eigen::Vector3d;

  namespace {
    // Minimum number of points to constrain the problem. See H&Z pg. 179.
    const int kNumLandmarks = 10000; /* lot of points so that enough are visible */
    const int kImageWidth = 1920;
    const int kImageHeight = 1080;
    const double kVerticalFov = D2R(90.0);

    // Bounding volume for 3D points.
    const double kMinX = -2.0;
    const double kMinY = -2.0;
    const double kMinZ = -2.0;
    const double kMaxX = 2.0;
    const double kMaxY = 2.0;
    const double kMaxZ = 2.0;

    const unsigned int kDescriptorLength = 64;

    // Makes a default set of camera intrinsic parameters.
    CameraIntrinsics DefaultIntrinsics() {
      CameraIntrinsics intrinsics;
      intrinsics.SetImageLeft(0);
      intrinsics.SetImageTop(0);
      intrinsics.SetImageWidth(kImageWidth);
      intrinsics.SetImageHeight(kImageHeight);
      intrinsics.SetVerticalFOV(kVerticalFov);
      intrinsics.SetFU(intrinsics.f_v());
      intrinsics.SetCU(0.5 * kImageWidth);
      intrinsics.SetCV(0.5 * kImageHeight);

      return intrinsics;
    }

    // Makes a random 3D point.
    Point3D RandomPoint() {
      CHECK(kMinX <= kMaxX);
      CHECK(kMinY <= kMaxY);
      CHECK(kMinZ <= kMaxZ);

      static math::RandomGenerator rng(0);
      const double x = rng.DoubleUniform(kMinX, kMaxX);
      const double y = rng.DoubleUniform(kMinY, kMaxY);
      const double z = rng.DoubleUniform(kMinZ, kMaxZ);
      return Point3D(x, y, z);
    }

    // Creates one observation for each landmark in the scene, and adds them to the
    // view. Also creates 'num_bad_matches' bad observations (of random 3D points).
    // Returns indices of landmarks that were successfully projected.
    std::vector<LandmarkIndex> CreateObservations(
        const std::vector<LandmarkIndex>& landmark_indices,
        ViewIndex view_index, unsigned int num_bad_matches) {
      // Get view.
      View::Ptr view = View::GetView(view_index);
      CHECK_NOTNULL(view.get());

      // For each landmark that projects into the view, create an observation and
      // add it to the view.
      std::vector<LandmarkIndex> projected_landmarks;
      for (const auto& landmark_index : landmark_indices) {
        Landmark::Ptr landmark = Landmark::GetLandmark(landmark_index);
        CHECK_NOTNULL(landmark.get());

        double u = 0.0, v = 0.0;
        const Point3D point = landmark->Position();
        if (!view->Camera().WorldToImage(point.X(), point.Y(), point.Z(), &u, &v))
          continue;

        // Creating the observation automatically adds it to the view.
        Observation::Ptr observation =
          Observation::Create(view, Feature(u, v), landmark->Descriptor());

        projected_landmarks.push_back(landmark_index);
      }

      // Make some bad observations also.
      for (unsigned int ii = 0; ii < num_bad_matches; ++ii) {
        double u = 0.0, v = 0.0;
        Point3D point = RandomPoint();
        if (!view->Camera().WorldToImage(point.X(), point.Y(), point.Z(), &u, &v))
          continue;

        // Creating the observation automatically adds it to the view.
        Observation::Ptr observation =
          Observation::Create(view, Feature(u, v), Descriptor::Random(kDescriptorLength));
      }

      return projected_landmarks;
    }

    void TestMatcher(unsigned int num_bad_matches) {
      // Clean up from other tests.
      Landmark::ResetLandmarks();
      View::ResetViews();

      // Make a camera with a random translation and rotation.
      Camera camera;
      CameraExtrinsics extrinsics;
      const Point3D camera_pose = RandomPoint();
      const Vector3d euler_angles(Vector3d::Random() * D2R(180.0));
      extrinsics.Translate(camera_pose.X(), camera_pose.Y(), camera_pose.Z());
      extrinsics.Rotate(EulerAnglesToMatrix(euler_angles));
      camera.SetExtrinsics(extrinsics);
      camera.SetIntrinsics(DefaultIntrinsics());

      // Initialize a view for this camera.
      View::Ptr view = View::Create(camera);

      // Make a bunch of randomly positioned landmarks.
      for (unsigned int ii = 0; ii < kNumLandmarks; ++ii) {
        Point3D point = RandomPoint();
        Landmark::Ptr landmark = Landmark::Create();
        landmark->SetPosition(point);
        landmark->SetDescriptor(Descriptor::Random(kDescriptorLength));
      }
      std::vector<LandmarkIndex> landmark_indices =
        Landmark::ExistingLandmarkIndices();

      // Create observations of the landmarks (no bad observations).
      std::vector<LandmarkIndex> projected_landmarks =
        CreateObservations(landmark_indices, view->Index(), num_bad_matches);
      ASSERT_LT(0, projected_landmarks.size());

      // Make sure the distance metric (which is global across tests) is set up
      // correctly.
      DistanceMetric& distance = DistanceMetric::Instance();
      distance.SetMetric(DistanceMetric::Metric::SCALED_L2);
      distance.SetMaximumDistance(std::numeric_limits<double>::max());

      // Match 2D features in the view to 3D landmarks.
      FeatureMatcherOptions options;
      options.min_num_feature_matches = projected_landmarks.size();
      FastMatcher2D3D feature_matcher;
      EXPECT_TRUE(feature_matcher.Match(options, view->Index(), landmark_indices));

      // Iterate over all observations in the view and check that they are correctly
      // matched.
      std::vector<Observation::Ptr> observations = view->Observations();
      for (size_t ii = 0; ii < observations.size(); ++ii) {
        if (!observations[ii]->IsMatched())
          continue;

        EXPECT_FALSE(observations[ii]->IsIncorporated());
        EXPECT_EQ(projected_landmarks[ii], observations[ii]->GetLandmark()->Index());
      }

      // Clean up.
      Landmark::ResetLandmarks();
      View::ResetViews();
    }

  }  //\namespace

  // Test with 1 to 1 correspondence between observations in the view and existing
  // landmarks.
  TEST(FastMatcher2D3D, TestFastMatcher2D3DNoiseless) {
    TestMatcher(0);
  }

  // Test with many to 1 correspondence between observations in the view and existing
  // landmarks.
  TEST(FastMatcher2D3D, TestFastMatcher2D3DNoisy) {
    TestMatcher(0.5 * kNumLandmarks);
  }

  // Test with MANY to 1 correspondence between observations in the view and existing
  // landmarks.
  TEST(FastMatcher2D3D, TestFastMatcher2D3DVeryNoisy) {
    TestMatcher(1.0 * kNumLandmarks);
  }


}  //\namespace bsfm
