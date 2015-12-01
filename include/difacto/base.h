#pragma once
#include <string>
#include "ps/base.h"

namespace difacto {

/*!
 * \brief use float as the weight and gradient type
 */
typedef float real_t;

/*!
 * \brief use PS's key (often uint64_t) as the feature index type
 */
typedef ps::Key feaid_t;

/**
 * \brief a list of keyword arguments
 */
typedef std::vector<std::pair<std::string, std::string>> KWArgs;

}  // namespace difacto
