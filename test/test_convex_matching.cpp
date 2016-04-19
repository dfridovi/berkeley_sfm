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
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 *          Erik Nelson            ( eanelson@eecs.berkeley.edu )
 */

#include <camera/camera.h>
#include <camera/camera_extrinsics.h>
#include <camera/camera_intrinsics.h>
#include <geometry/eight_point_algorithm_solver.h>
#include <geometry/rotation.h>
#include <image/drawing_utils.h>
#include <image/image.h>
#include <matching/descriptor_extractor.h>
#include <matching/distance_metric.h>
#include <matching/feature_match.h>
#include <matching/keypoint_detector.h>
#include <matching/naive_matcher_2d2d.h>
#include <matching/pairwise_image_match.h>
#include <math/random_generator.h>
#include <ransac/fundamental_matrix_ransac_problem.h>
#include <ransac/ransac.h>
#include <ransac/ransac_options.h>
#include <strings/join_filepath.h>

#include <gtest/gtest.h>

DEFINE_string(convex_matched_image1, "lion1.jpg",
              "Name of the first image used to test convex feature matching.");
DEFINE_string(convex_matched_image2, "lion2.jpg",
              "Name of the second image used to test convex feature matching.");
DEFINE_string(matrix_file, "A.csv",
              "Name of the file containing the SPD matrix for warped L2 matching.");
DEFINE_bool(convex_draw_feature_matches, true,
            "If true, open a window displaying raw (noisy) feature matches and "
            "post-RANSAC inlier matches.");

namespace bsfm {

using Eigen::Matrix3d;
using Eigen::Vector3d;

class TestConvexRansac : public ::testing::Test {
 protected:
  const std::string matrix_file = strings::JoinFilepath(
      BSFM_TEST_DATA_DIR, FLAGS_matrix_file.c_str());
  const std::string test_image1 = strings::JoinFilepath(
      BSFM_TEST_DATA_DIR, FLAGS_convex_matched_image1.c_str());
  const std::string test_image2 = strings::JoinFilepath(
      BSFM_TEST_DATA_DIR, FLAGS_convex_matched_image2.c_str());

 private:
  const int kImageWidth_ = 1920;
  const int kImageHeight_ = 1080;
  const double kVerticalFov_ = 0.5 * M_PI;
};  //\class TestConvexRansac

TEST_F(TestConvexRansac, TestDrawInliers) {

  // Load two images and compute feature matches. Draw matches with and without
  // RANSAC.
  if (!FLAGS_convex_draw_feature_matches) {
    return;
  }

  // Get some noisy feature matches.
  Image image1(test_image1.c_str());
  Image image2(test_image2.c_str());

  image1.Resize(0.25);
  image2.Resize(0.25);

  KeypointDetector detector;
  detector.SetDetector("SURF");

  KeypointList keypoints1;
  KeypointList keypoints2;
  detector.DetectKeypoints(image1, keypoints1);
  detector.DetectKeypoints(image2, keypoints2);

  // DistanceMetric::Instance().SetMetric(DistanceMetric::Metric::SCALED_L2);
  DescriptorExtractor extractor;
  extractor.SetDescriptor("SURF");

  std::vector<Feature> features1;
  std::vector<Feature> features2;
  std::vector<Descriptor> descriptors1;
  std::vector<Descriptor> descriptors2;
  extractor.DescribeFeatures(image1, keypoints1, features1, descriptors1);
  extractor.DescribeFeatures(image2, keypoints2, features2, descriptors2);

  FeatureMatcherOptions matcher_options;
  matcher_options.lowes_ratio = 1.1;
  matcher_options.min_num_feature_matches = 8;
  matcher_options.distance_metric = "WARPED_L2";
  matcher_options.matrix_file = matrix_file;
  NaiveMatcher2D2D feature_matcher;
  feature_matcher.AddImageFeatures(features1, descriptors1);
  feature_matcher.AddImageFeatures(features2, descriptors2);
  PairwiseImageMatchList image_matches;
  feature_matcher.MatchImages(matcher_options, image_matches);

  ASSERT_TRUE(image_matches.size() == 1);

  // RANSAC the feature matches to get inliers.
  FundamentalMatrixRansacProblem problem;
  problem.SetData(image_matches[0].feature_matches_);

  // Create the ransac solver, set options, and run RANSAC on the problem.
  Ransac<FeatureMatch, FundamentalMatrixRansacModel> solver;
  RansacOptions ransac_options;

  ransac_options.iterations = 5000;
  ransac_options.acceptable_error = 0.1;
  ransac_options.minimum_num_inliers = 20;
  ransac_options.num_samples = 8;

  solver.SetOptions(ransac_options);
  solver.Run(problem);

  ASSERT_TRUE(problem.SolutionFound());

  drawing::DrawImageFeatureMatches(image1, image2,
                                   image_matches[0].feature_matches_,
                                   "Noisy Matched Features");

  const FeatureMatchList& inliers = problem.Inliers();
  drawing::DrawImageFeatureMatches(image1, image2, inliers,
                                   "Inlier Matched Features");
}

}  //\namespace bsfm
