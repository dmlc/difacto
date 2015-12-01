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
 * \brief a keyword argument used for initializaiton
 */
typedef std::par<std::string, std::string> KWArg;

/**
 * \brief a list of keyword arguments
 */
typedef std::vector<KVArg> KWArgs;

}  // namespace difacto
