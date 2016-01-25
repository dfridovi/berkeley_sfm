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

#include <image/drawing_utils.h>
#include <image/image.h>
#include <matching/descriptor_extractor.h>
#include <matching/distance_metric.h>
#include <matching/keypoint_detector.h>
#include <matching/naive_matcher_2d2d.h>
#include <strings/join_filepath.h>
#include <util/types.h>

#include <gflags/gflags.h>
#include <gtest/gtest.h>

DEFINE_string(matched_image1, "campanile_view1.jpg",
              "Name of the first image used to test feature matching.");
DEFINE_string(matched_image2, "campanile_view2.jpg",
              "Name of the second image used to test feature matching.");
DEFINE_bool(draw_feature_matches, false,
            "If true, open a window displaying feature matches for each test.");

namespace bsfm {

class TestNaiveMatcher2D2D : public ::testing::Test {
 protected:
  const std::string test_image1 = strings::JoinFilepath(
      BSFM_TEST_DATA_DIR, FLAGS_matched_image1.c_str());
  const std::string test_image2 = strings::JoinFilepath(
      BSFM_TEST_DATA_DIR, FLAGS_matched_image2.c_str());

  const unsigned int expected_matched_features_symmetric_floating = 199;
  const unsigned int expected_matched_features_asymmetric_floating = 480;
  const unsigned int expected_matched_features_symmetric_binary = 806;
  const unsigned int expected_matched_features_asymmetric_binary = 2501;
};  //\class TestNaiveFeatureMatcher

TEST_F(TestNaiveMatcher2D2D, TestNaiveMatcherSiftSift) {
  // Load two images.
  Image image1(test_image1.c_str());
  Image image2(test_image2.c_str());

  image1.Resize(640.0, 480.0);
  image2.Resize(640.0, 480.0);
  image1.RotateClockwise();
  image2.RotateClockwise();

  FeatureMatcherOptions options;
  options.distance_metric = "SCALED_L2";
  options.threshold_image_distance = false;
  NaiveMatcher2D2D feature_matcher;

  KeypointDetector detector;
  detector.SetDetector("SIFT");

  KeypointList keypoints1;
  KeypointList keypoints2;
  detector.DetectKeypoints(image1, keypoints1);
  detector.DetectKeypoints(image2, keypoints2);
  LOG(INFO) << "Detected " << keypoints1.size() << " keypoints from image 1.";
  LOG(INFO) << "Detected " << keypoints2.size() << " keypoints from image 2.";

  // DistanceMetric::Instance().SetMetric(DistanceMetric::Metric::SCALED_L2);
  DescriptorExtractor extractor;
  extractor.SetDescriptor("SIFT");

  std::vector<Feature> features1;
  std::vector<Feature> features2;
  std::vector<Descriptor> descriptors1;
  std::vector<Descriptor> descriptors2;
  extractor.DescribeFeatures(image1, keypoints1, features1, descriptors1);
  extractor.DescribeFeatures(image2, keypoints2, features2, descriptors2);
  LOG(INFO) << "Extracted " << features1.size() << " features from image 1.";
  LOG(INFO) << "Extracted " << features2.size() << " features from image 2.";

  feature_matcher.AddImageFeatures(features1, descriptors1);
  feature_matcher.AddImageFeatures(features2, descriptors2);

  // Match images with symmetric feature matches enforced.
  PairwiseImageMatchList image_matches;
  ASSERT_TRUE(feature_matcher.MatchImages(options, image_matches));

  // Different OpenCV versions will yield different numbers of
  // keypoints/descriptors. Check for matches with some tolerance.
  EXPECT_NEAR(expected_matched_features_symmetric_floating,
              image_matches[0].feature_matches_.size(),
              static_cast<size_t>(0.9 *
                 static_cast<double>(expected_matched_features_symmetric_floating)));


  // Draw feature matches.
  if (FLAGS_draw_feature_matches) {
    drawing::DrawImageFeatureMatches(image1, image2,
                                     image_matches[0].feature_matches_,
                                     "Naive Symmetric Matched Features");
  }

  // Match images without enforcing symmetric feature matches.
  image_matches = PairwiseImageMatchList();
  options.require_symmetric_matches = false;
  ASSERT_TRUE(feature_matcher.MatchImages(options, image_matches));

  // Different OpenCV versoins will yield different numbers of
  // keypoints/descriptors. Check for matches with some tolerance.
  EXPECT_NEAR(expected_matched_features_asymmetric_floating,
              image_matches[0].feature_matches_.size(),
              static_cast<size_t>(0.9 *
                 static_cast<double>(expected_matched_features_asymmetric_floating)));

  // Draw feature matches.
  if (FLAGS_draw_feature_matches) {
    drawing::DrawImageFeatureMatches(image1, image2,
                                     image_matches[0].feature_matches_,
                                     "Asymmetric Matched Features");
  }
}

TEST_F(TestNaiveMatcher2D2D, TestNaiveMatcherFastOrb) {
  // Load two images.
  Image image1(test_image1.c_str());
  Image image2(test_image2.c_str());

  image1.Resize(640.0, 480.0);
  image2.Resize(640.0, 480.0);
  image1.RotateClockwise();
  image2.RotateClockwise();

  KeypointDetector detector;
  detector.SetDetector("FAST");

  KeypointList keypoints1;
  KeypointList keypoints2;
  detector.DetectKeypoints(image1, keypoints1);
  detector.DetectKeypoints(image2, keypoints2);
  LOG(INFO) << "Detected " << keypoints1.size() << " keypoints from image 1.";
  LOG(INFO) << "Detected " << keypoints2.size() << " keypoints from image 2.";

  DescriptorExtractor extractor;
  extractor.SetDescriptor("ORB");

  std::vector<Feature> features1;
  std::vector<Feature> features2;
  std::vector<Descriptor> descriptors1;
  std::vector<Descriptor> descriptors2;
  extractor.DescribeFeatures(image1, keypoints1, features1, descriptors1);
  extractor.DescribeFeatures(image2, keypoints2, features2, descriptors2);
  LOG(INFO) << "Extracted " << features1.size() << " features from image 1.";
  LOG(INFO) << "Extracted " << features2.size() << " features from image 2.";

  FeatureMatcherOptions options;
  options.threshold_image_distance = false;
  options.distance_metric = "HAMMING";
  NaiveMatcher2D2D feature_matcher;

  feature_matcher.AddImageFeatures(features1, descriptors1);
  feature_matcher.AddImageFeatures(features2, descriptors2);

  // Match images with symmetric feature matches enforced.
  PairwiseImageMatchList image_matches;
  ASSERT_TRUE(feature_matcher.MatchImages(options, image_matches));

  // Different OpenCV versions will yield different numbers of
  // keypoints/descriptors. Check for matches with some tolerance.
  EXPECT_NEAR(expected_matched_features_symmetric_binary,
              image_matches[0].feature_matches_.size(),
              static_cast<size_t>(0.9 *
                 static_cast<double>(expected_matched_features_symmetric_binary)));

  // Draw feature matches.
  if (FLAGS_draw_feature_matches) {
    drawing::DrawImageFeatureMatches(image1, image2,
                                     image_matches[0].feature_matches_,
                                     "Symmetric Matched Features");
  }

  // Match images without enforcing symmetric feature matches.
  image_matches = PairwiseImageMatchList();
  options.require_symmetric_matches = false;
  ASSERT_TRUE(feature_matcher.MatchImages(options, image_matches));

  // Different OpenCV versoins will yield different numbers of
  // keypoints/descriptors. Check for matches with some tolerance.
  EXPECT_NEAR(expected_matched_features_asymmetric_binary,
              image_matches[0].feature_matches_.size(),
              static_cast<size_t>(0.9 *
                 static_cast<double>(expected_matched_features_asymmetric_binary)));


  // Draw feature matches.
  if (FLAGS_draw_feature_matches) {
    drawing::DrawImageFeatureMatches(image1, image2,
                                     image_matches[0].feature_matches_,
                                     "Naive Asymmetric Matched Features");
  }
}

}  //\namespace bsfm
