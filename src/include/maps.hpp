#ifndef BUILD_SRC_INCLUDE_MAPS_HPP_
#define BUILD_SRC_INCLUDE_MAPS_HPP_
#include "params.hpp"
#include "view.hpp"
#include "types.hpp"
#include <utils.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/opencv.hpp"

#include "maps.hpp"

namespace MonocularVO
{

  using FrameSharedPtr = std::shared_ptr<Frame>;
  using Batch = std::vector<FrameSharedPtr>;

  using MatchKeyFrameSharedPtr = std::shared_ptr<MatchKeyFrame>;


class MapInitial
{
public:
  MonocularVO::Params params_;
  using MapInitialSharedPtr = std::shared_ptr<MapInitial>;

  explicit MapInitial(MonocularVO::Params  params);

  // all frames
  std::vector<FrameSharedPtr> frames_;

  int count_local_landmark_{};

  FrameSharedPtr
  get_last_key_frame();

  void
  push_frame(const MonocularVO::FrameSharedPtr& frame);

  MonocularVO::FrameSharedPtr
  get_curr_frame();

  MonocularVO::FrameSharedPtr
  get_prev_frame();

  int
  get_frame_count();

  void
  delete_past_images();
  void
  update_past_frames_optical_flow(const std::vector<uchar>& status);

  void
  update_past_frames_epipolar(const cv::Mat& inliers_F,
                              const cv::Mat& inliers_E);

  void
  print_frames_info();

  int
  get_key_frame_count();

  int
  get_last_key_frame_id_in_frames();

  int
  get_prev_key_frame_id_in_frames();

  Batch build_batch_1();

  Batch build_batch_2();



/*
  void
  add_match(const MatchKeyFrameSharedPtr & match_key_frame)
  {
    vector_match_keyframe.push_back(match_key_frame);
    std::cout << "New keyframe match is added to map." << std::endl;
*/
/*
    for (const auto & match:vector_match_keyframe)
    {
        std::cout << "Old key-frame id: " << match->old_frame_id <<
        " | New key-frame id: " << match->new_frame_id << std::endl;
    }
*//*


      build_observations();
  }

  int
  build_observations()
  {
    int total_collected_landmark_count = 0;

    if (vector_match_keyframe.size() >= match_count_to_start_)
    {
      std::cout << "Building observations:" << std::endl;
      int view_count = vector_match_keyframe.size()+1;
      std::cout << "Total view count: " << view_count << std::endl;
      MatchKeyFrameSharedPtr curr_match = vector_match_keyframe.back();
      for (int i=0; i<curr_match->old_kpt_ids.size(); i++)
      {

        int query_kpt_new = curr_match->new_kpt_ids[i];
        int query_kpt_old = curr_match->old_kpt_ids[i];

        Landmark landmark;
        landmark.observed_frame_ids.push_back(curr_match->new_frame_id);
        landmark.pxs.push_back(curr_match->new_keyframe_kpts.at(query_kpt_new).pt);
        landmark.observed_frame_ids.push_back(curr_match->old_frame_id);
        landmark.pxs.push_back(curr_match->old_keyframe_kpts.at(query_kpt_old).pt);

        //std::cout << "Query for keypoint: " << i << " observed in "  << curr_match->new_frame_id << ".frame" <<  std::endl;
        //std::cout << "Query for keypoint: " << i << " observed in "  << curr_match->old_frame_id << ".frame" <<  std::endl;

        int seen_count = 2;


        //std::cout << "Curr match new id: " << curr_match->new_frame_id << " | old id: " << curr_match->old_frame_id << std::endl;
        for (int j=0; j<vector_match_keyframe.size(); j++)
        {
          if (j<vector_match_keyframe.size()-1)
          {
            MatchKeyFrameSharedPtr prev_match = vector_match_keyframe.at(vector_match_keyframe.size()-(j+2));
            //std::cout << "Prev match new id: " << prev_match->new_frame_id << " | old id: " << prev_match->old_frame_id << std::endl;

            // Iterate over prev match ids:
            bool match_exist = false;
            int search_kpt_old;
            for (int k=0; k<prev_match->new_kpt_ids.size(); k++)
            {
              int search_kpt_new = prev_match->new_kpt_ids[k];
              if (query_kpt_old == search_kpt_new)
              {
                search_kpt_old = prev_match->old_kpt_ids[k];
                match_exist = true;
                landmark.observed_frame_ids.push_back(prev_match->old_frame_id);
                landmark.pxs.push_back(prev_match->old_keyframe_kpts.at(search_kpt_old).pt);
                //std::cout << "Query for keypoint: " << i << " observed in "  << prev_match->old_frame_id << ".frame" <<  std::endl;
                seen_count++;
                break;
              }
            }

            if(!match_exist)
              break;

          }
        }
        if (seen_count == view_count)
        {
          total_collected_landmark_count++;
          //std::cout << "Query for keypoint " << i << " is observed all previous frames." << std::endl;
          //std::cout << "**********************************" << std::endl;
        }


      }
      std::cout << "Total builded observation count: " << total_collected_landmark_count << std::endl;
      return total_collected_landmark_count;
    } else
      return -1;
  }
*/

private:

}; // eof MapInitial


// #######################################################
// #######################################################
// #######################################################
// #######################################################
// #######################################################
// #######################################################
// #######################################################
// #######################################################
// #######################################################
// #######################################################
// #######################################################


class MapLocal
{
public:
  using FrameSharedPtr = std::shared_ptr<Frame>;
  using Batch = std::vector<FrameSharedPtr>;

  explicit MapLocal();

  void print_curr_batch_info();
  void push_key_frames(const Batch& batch_curr_);
  FrameSharedPtr get_curr_key_frame();
  FrameSharedPtr get_prev_key_frame();
  void solve(MapLocal::Batch& batch);

private:
  std::vector<FrameSharedPtr> key_frames_;




};




}

#endif  // BUILD_SRC_INCLUDE_MAPS_HPP_
