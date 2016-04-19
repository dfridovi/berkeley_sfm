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

#include "feature_matcher.h"
#include <iostream>

namespace bsfm {

// Append features from a single image to the list of all image features.
void FeatureMatcher::AddImageFeatures(
    const std::vector<Feature>& image_features,
    const std::vector<Descriptor>& image_descriptors) {
  image_features_.push_back(image_features);
  image_descriptors_.push_back(image_descriptors);
}

// Append features from a set of images to the list of all image features.
void FeatureMatcher::AddImageFeatures(
    const std::vector<std::vector<Feature> >& image_features,
    const std::vector<std::vector<Descriptor> >& image_descriptors) {
  image_features_.insert(image_features_.end(), image_features.begin(),
                         image_features.end());
  image_descriptors_.insert(image_descriptors_.end(), image_descriptors.begin(),
                            image_descriptors.end());
}

bool FeatureMatcher::MatchImages(const FeatureMatcherOptions& options,
                                 PairwiseImageMatchList& image_matches) {
  // Store the matching options locally.
  options_ = options;

  // Iterate over all pairs of images and attempt to match them together by
  // comparing their features.
  for (size_t ii = 0; ii < image_features_.size(); ++ii) {
    // Make sure this image has features.
    if (image_features_[ii].size() == 0) {
      continue;
    }
    for (size_t jj = ii + 1; jj < image_features_.size(); ++jj) {
      // Make sure this image has features.
      if (image_features_[jj].size() == 0) {
        continue;
      }

      // Create an image match object and attempt to match image ii to image jj.
      PairwiseImageMatch image_match;
      if (!MatchImagePair(ii, jj, image_match)) {
        VLOG(1) << "Could not match image " << ii << " to image " << jj << ".";
        std::cout << "oops" << std::endl;
        continue;
      }

      // If the image match was successful, store it.
      image_matches.push_back(image_match);
    }
  }

  // Return whether or not any of the image pairs were matched.
  return image_matches.size() > 0;
}

void FeatureMatcher::SymmetricMatches(
    const std::vector<LightFeatureMatch>& feature_matches_lhs,
    std::vector<LightFeatureMatch>& feature_matches_rhs) {
  std::unordered_map<int, int> feature_indices;
  feature_indices.reserve(feature_matches_lhs.size());

  // Add all LHS matches to the map.
  for (const auto& feature_match : feature_matches_lhs) {
    feature_indices.insert(std::make_pair(feature_match.feature_index1_,
                                          feature_match.feature_index2_));
  }

  // For each match in the RHS set, search for the same match in the LHS set.
  // If the match is not symmetric, remove it from the RHS set.
  auto rhs_iter = feature_matches_rhs.begin();
  while (rhs_iter != feature_matches_rhs.end()) {
    const auto& lhs_matched_iter =
        feature_indices.find(rhs_iter->feature_index2_);

    // If a symmetric match is found, keep it in the RHS set.
    if (lhs_matched_iter != feature_indices.end()) {
      if (lhs_matched_iter->second == rhs_iter->feature_index1_) {
        ++rhs_iter;
        continue;
      }
    }

    // Remove the non-symmetric match and continue on.
    feature_matches_rhs.erase(rhs_iter);
  }
}

}  //\namespace bsfm
