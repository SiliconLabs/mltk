#pragma once 

#include "logging/logger.hpp"


#ifndef MLTK_LOG_LEVEL
#define MLTK_LOG_LEVEL 0
#endif

#if MLTK_LOG_LEVEL <= 0
#define MLTK_DEBUG(msg, ...) ::mltk::get_logger().debug(msg, ## __VA_ARGS__)
#endif
#if MLTK_LOG_LEVEL <= 1
#define MLTK_INFO(msg, ...) ::mltk::get_logger().info(msg, ## __VA_ARGS__)
#endif
#if MLTK_LOG_LEVEL <= 2
#define MLTK_WARN(msg, ...) ::mltk::get_logger().warn(msg, ## __VA_ARGS__)
#endif
#if MLTK_LOG_LEVEL <= 3
#define MLTK_ERROR(msg, ...) ::mltk::get_logger().error(msg, ## __VA_ARGS__)
#endif


#ifndef MLTK_DEBUG
#define MLTK_DEBUG(...)
#endif
#ifndef MLTK_INFO
#define MLTK_INFO(...)
#endif
#ifndef MLTK_WARN
#define MLTK_WARN(...)
#endif
#ifndef MLTK_ERROR
#define MLTK_ERROR(...)
#endif


namespace mltk
{

using Logger = logging::Logger;
using LogLevel = logging::Level;


Logger& get_logger();
bool set_log_level(LogLevel level);


} // namespace mltk