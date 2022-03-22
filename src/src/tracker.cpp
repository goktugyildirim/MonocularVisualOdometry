//
// Created by goktug on 22.03.2022.
//

#include "tracker.hpp"

namespace MonocularVO
{

Tracker::Tracker(const MonocularVO::Params& params)
:m_params(params)
{

}

void
Tracker::track_observations(const std::vector<cv::Point2f>& prev_frame_kpts,
                            const std::vector<cv::Point2f>& curr_frame_kpts,
                            const int& frame_id)
{



}



}
