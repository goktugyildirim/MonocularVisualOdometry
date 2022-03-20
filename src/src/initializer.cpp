//
// Created by goktug on 20.03.2022.
//

#include "initializer.hpp"

namespace MonocularVO
{
Initializer::Initializer(const MonocularVO::Params& params)
: m_params(params)
{}
void Initializer::try_init(const FrameSharedPtr &ref_frame,
                           const FrameSharedPtr &curr_frame,
                           std::vector<int> &tracked_p2d_ids)
{
  std::cout << "Trying to init..." << std::endl;
  std::cout << "Count ref frame tracked point count before initialization:" <<
    tracked_p2d_ids.size() << std::endl;
}

}
