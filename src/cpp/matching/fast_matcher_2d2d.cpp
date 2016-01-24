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

#include "fast_matcher_2d2d.h"
#include "../sfm/flann_descriptr_kdtree.h"

namespace bsfm {

  // Match two images together by doing a nearest neighbor search.
  bool FastMatcher2D2D::MatchImagePair(int image_index1, int image_index2,
                                       PairwiseImageMatch& image_match) {
    image_match.feature_matches_.clear();
    image_match.descriptor_indices1_.clear();
    image_match.descriptor_indices2_.clear();

    // Get the descriptors corresponding to these two images.
    std::vector<Descriptor>& descriptors1 = image_descriptors_[image_index1];
    std::vector<Descriptor>& descriptors2 = image_descriptors_[image_index2];

    // Normalize descriptors if required by the distance metric.
    DistanceMetric::Instance().SetMetric(options_.distance_metric);
    DistanceMetric::Instance().MaybeNormalizeDescriptors(descriptors1);
    DistanceMetric::Instance().MaybeNormalizeDescriptors(descriptors2);

    // Compute forward (and reverse, if applicable) matches.
    LightFeatureMatchList light_feature_matches;
    ComputePutativeMatches(descriptors1, descriptors2, light_feature_matches);

    // Check that we got enough matches here. If we didn't, reverse matches won't
    // help us.
    if (light_feature_matches.size() < options_.min_num_feature_matches) {
      return false;
    }

    if (options_.require_symmetric_matches) {
      LightFeatureMatchList reverse_light_feature_matches;
      ComputePutativeMatches(descriptors2, descriptors1,
                             reverse_light_feature_matches);
      SymmetricMatches(reverse_light_feature_matches, light_feature_matches);
    }

    if (light_feature_matches.size() < options_.min_num_feature_matches) {
      return false;
    }

    // Check how many features the user wants.
    unsigned int num_features_out = light_feature_matches.size();
    if (options_.only_keep_best_matches) {
      num_features_out =
        std::min(num_features_out, options_.num_best_matches);

      // Return relevant matches in sorted order.
      std::partial_sort(light_feature_matches.begin(),
                        light_feature_matches.begin() + num_features_out,
                        light_feature_matches.end(),
                        LightFeatureMatch::SortByDistance);
    }

    // Convert from LightFeatureMatchList to FeatureMatchList for the output.
    for (int ii = 0; ii < num_features_out; ++ii) {
      const auto& match = light_feature_matches[ii];

      const Feature& matched_feature1 =
        image_features_[image_index1][match.feature_index1_];
      const Feature& matched_feature2 =
        image_features_[image_index2][match.feature_index2_];

      // Check that the features are close in image space.
      if (options_.threshold_image_distance) {
        const double du = matched_feature1.u_ - matched_feature2.u_;
        const double dv = matched_feature1.v_ - matched_feature2.v_;
        if (std::sqrt(du*du + dv*dv) > options_.maximum_image_distance) {
          continue;
        }
      }
      image_match.feature_matches_.emplace_back(matched_feature1,
                                                matched_feature2);

      // Also store the index of each descriptor used for this match.
      image_match.descriptor_indices1_.push_back(match.feature_index1_);
      image_match.descriptor_indices2_.push_back(match.feature_index2_);
    }

    return true;
  }

  // Compute putative matches between feature descriptors for an image pair.
  // These might be removed later on due to e.g. not being symmetric, etc.
  void FastMatcher2D2D::ComputePutativeMatches(
                            const std::vector<Descriptor>& descriptors1,
                            const std::vector<Descriptor>& descriptors2,
                            std::vector<LightFeatureMatch>& putative_matches) {
    putative_matches.clear();

    // Get the singleton distance metric for descriptor comparison.
    DistanceMetric& distance = DistanceMetric::Instance();

    // Set the maximum tolerable distance between descriptors, if applicable.
    if (options_.enforce_maximum_descriptor_distance) {
      distance.SetMaximumDistance(options_.maximum_descriptor_distance);
    }

    // Create a kd-tree to store descriptors2.
    FlannDescriptorKDTree kdtree;
    kdtree.AddDescriptors(descriptors2);

    // Find two nearest neighbors for each descriptor in descriptors1 and apply
    // Lowe's ratio test.
    for (size_t ii = 0; ii < descriptors1.size(); ++ii) {
      std::vector<int> nn_indices;
      std::vector<double> nn_distances;
      const unsigned int kNearestNeighbors = 2;

      if (!kdtree.NearestNeighbors(descriptors1[ii], kNearestNeighbors,
                                   nn_indices, nn_distances)) {
        VLOG(1) << "k nearest neighbor search failed. Continuing.";
        continue;
      }

      // If max distance was not set above, distance.Max() will be infinity and
      // this will always be true.
      if (nn_distances[0] > distance.Max())
        continue;

      // The second best match must be within the lowes ratio of the best match.
      LightFeatureMatch match(ii, nn_indices[0], nn_distances[0]);
      if (options_.use_lowes_ratio) {
        if (nn_distances[0] < options_.lowes_ratio * nn_distances[1]) {
          putative_matches.emplace_back(match);
        }
      } else {
        putative_matches.emplace_back(match);
      }
    }
  }

}  //\namespace bsfm
