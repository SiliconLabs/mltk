
#include <new>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cassert>


#include "em_cmu.h"

#include "cpputils/helpers.hpp"
#include "cpputils/string.hpp"
#include "logging/logger.hpp"

#include "profiling/profiler.hpp"




#define PROFILER_WARN(msg, ...) get_logger().warn(msg, ## __VA_ARGS__)


using namespace cpputils;



namespace profiling
{


constexpr const char* const LINE_DIVIDER = "----------------------";


static void clear_visited_flags(Profiler *profiler);
static void print_profiler_statistics(Profiler *profiler, logging::Logger& logger, int level, char *level_str, bool pretty_print);
static void print_profiler_metric(Profiler *profiler, logging::Logger &logger, int level, char *level_str, bool pretty_print);
static logging::Logger& get_logger(logging::Logger* logger = nullptr);

CREATE_STATIC_OBJECT_BUFFER(ProfilerList, _profiler_list_buffer);
ProfilerList *_profilers;



/*************************************************************************************************/
void set_profiler_list(ProfilerList* list)
{
    _profilers = list;
}


/*************************************************************************************************/
#ifndef MLTK_DLL_IMPORT
ProfilerList& get_profilers()
{
    if(_profilers == nullptr)
    {
        _profilers = new(_profiler_list_buffer)ProfilerList();
    }
    return *_profilers;
}
#endif


/*************************************************************************************************/
template<typename T>
bool register_profiler_internal(const char* name, T* &profiler, Profiler* parent)
{
    if(strstr(name, Fullname::SEPARATOR) != nullptr)
    {
        assert(!"Profiler names must not contain ::");
        return false;
    }

    Fullname fullname;

    Fullname::create(fullname, name, parent);

    if(profiling::get(fullname.value) != nullptr)
    {
        PROFILER_WARN("Profiler: %s already registered, can't register again", fullname.value);
        return false;
    }

    const uint32_t alloc_size = T::required_alloc_size(name);
    uint8_t* ptr = static_cast<uint8_t*>(malloc(alloc_size));
    if(ptr == nullptr)
    {
        PROFILER_WARN("Profiler: %s, failed to alloc memory", fullname.value);
        return false;
    }

    char* name_ptr = (char*)(ptr + sizeof(T));
    strcpy(name_ptr, name);
    profiler = new(ptr)T(ptr, name_ptr);
    get_profilers().append(profiler);

    if(parent != nullptr)
    {
        if(!profiler->parent(parent))
        {
            profiler->unlink();
            return false;
        }
    }

    return true;
}

/*************************************************************************************************/
bool register_profiler(const char* name, Profiler* &profiler, Profiler* parent)
{
    return register_profiler_internal(name, profiler, parent);
}

/*************************************************************************************************/
bool register_profiler(const char* name, AveragedProfiler* &profiler, Profiler* parent)
{
    return register_profiler_internal(name, profiler, parent);
}

/*************************************************************************************************/
bool unregister(const Profiler* profiler, bool unregister_children)
{
    Fullname fullname;
    return (profiler == nullptr) ? false : unregister(profiler->fullname(fullname), unregister_children);
}

/*************************************************************************************************/
bool unregister(const char* name, bool unregister_children)
{
    auto profiler = get(name);
    if(profiler != nullptr)
    {
        auto parent = profiler->parent();
        if(parent != nullptr)
        {
            parent->_children.remove(profiler);
        }

        if(unregister_children)
        {
            auto& children = profiler->_children;

            while(children.size() > 0)
            {
                auto& child = children.first();
                unregister(child);
            }
        }

        get_profilers().remove(profiler);

        return true;
    }

    return false;
}

/*************************************************************************************************/
bool reset(const char* name, bool reset_children)
{
    return reset(profiling::get(name), reset_children);
}

/*************************************************************************************************/
bool reset(Profiler* profiler, bool reset_children)
{
    if(profiler == nullptr)
    {
        return false;
    }

    profiler->reset();


    if(reset_children)
    {
        for(auto child : profiler->_children)
        {
            reset(child, true);
        }
    }

    return true;
}

/*************************************************************************************************/
bool exists(const char* name)
{
    return (profiling::get(name) != nullptr);
}

/*************************************************************************************************/
Profiler* get(const char* name)
{
    if(name == nullptr)
    {
        assert(!"Null profiler name");
        return nullptr;
    }

    if(strstr(name, Fullname::SEPARATOR) != nullptr)
    {
        for(auto profiler : get_profilers())
        {
            Fullname fullname;
            if(strcmp(profiler->fullname(fullname), name) == 0)
            {
                return profiler;
            }
        }
    }
    else 
    {
        for(auto profiler : get_profilers())
        {
            if(strcmp(profiler->_name, name) == 0)
            {
                return profiler;
            }
        }
    }

    return nullptr;
}

/*************************************************************************************************/
void get_all(const char* name, cpputils::TypedList<Profiler*>& list)
{
    auto root = get(name);
    if(root == nullptr)
    {
        return;
    }

    std::function<void(profiling::Profiler*)> add_profilers;

    add_profilers = [&list, &add_profilers](Profiler* parent) 
    {
        list.append(parent);
        for(auto child : parent->children())
        {
            add_profilers(child);
        }
    };

    add_profilers(root);
}

/*************************************************************************************************/
void print_metrics(const char* name, logging::Logger *logger, bool pretty_print)
{
    print_metrics(profiling::get(name), logger, pretty_print);
}

/*************************************************************************************************/
void print_metrics(const Profiler* profiler, logging::Logger *optional_logger, bool pretty_print)
{
    if(profiler == nullptr)
    {
        return;
    }

    clear_visited_flags(const_cast<Profiler*>(profiler));

    char level_str[64];
    auto& logger = get_logger(optional_logger);

    if(pretty_print)
    {
        logger.info(LINE_DIVIDER);
    }
    logger.info("Profiler Metrics:");
    if(pretty_print)
    {
        logger.info(LINE_DIVIDER);
    }
    logger.info("CPU clock: %sHz", cpputils::format_units(CMU_ClockFreqGet(cmuClock_CORE), 1));

    print_profiler_metric(const_cast<Profiler*>(profiler), logger, 0, level_str, pretty_print);

    logger.info(LINE_DIVIDER);
}

/*************************************************************************************************/
void print_stats(const char* name, logging::Logger *logger, bool pretty_print)
{
    print_stats(profiling::get(name), logger, pretty_print);
}

/*************************************************************************************************/
void print_stats(const Profiler* profiler, logging::Logger *optional_logger, bool pretty_print)
{
    if(profiler == nullptr)
    {
        return;
    }

    clear_visited_flags(const_cast<Profiler*>(profiler));

    char level_str[64];
    auto& logger = get_logger(optional_logger);

    if(pretty_print)
    {
        logger.info(LINE_DIVIDER);
    }

    print_profiler_statistics(const_cast<Profiler*>(profiler), logger, 0, level_str, pretty_print);

    if(pretty_print)
    {
        logger.info(LINE_DIVIDER);
    }
}


/*************************************************************************************************/
uint32_t calculate_total_children_cpu_cycles(const Profiler* profiler, bool is_root)
{
    uint32_t total_cycles = 0;

    if(!is_root && !profiler->flags().isSet(Flag::ExcludeFromTotalChildrenCyclesReport))
    {
        const auto& s = profiler->stats();
        total_cycles += s.cpu_cycles;
    }
    for(auto child : profiler->_children)
    {
        total_cycles += calculate_total_children_cpu_cycles(child, false);
    }

    return total_cycles;
}

/*************************************************************************************************/
uint32_t calculate_total_children_accelerator_cycles(const Profiler* profiler, bool is_root)
{
    uint32_t total_cycles = 0;

    if(!is_root && !profiler->flags().isSet(Flag::ExcludeFromTotalChildrenCyclesReport))
    {
        const auto& s = profiler->stats();
        total_cycles += s.accelerator_cycles;
    }
    for(auto child : profiler->_children)
    {
        total_cycles += calculate_total_children_accelerator_cycles(child, false);
    }

    return total_cycles;
}



/** --------------------------------------------------------------------------------------------
 *  Internal functions
 * -------------------------------------------------------------------------------------------- **/

/*************************************************************************************************/
static void clear_visited_flags(Profiler *profiler)
{
    profiler->_was_visited = false;
    for(auto child : profiler->_children)
    {
        clear_visited_flags(child);
    }
}

/*************************************************************************************************/
static void print_profiler_statistics(
    Profiler *profiler, 
    logging::Logger& logger, 
    int level, 
    char *level_str, 
    bool pretty_print
)
{
    const auto& stats = profiler->stats(); 
    const auto metrics = profiler->metrics_including_children();
    const auto& flags = profiler->flags();
    bool printed_name = false;

    if(pretty_print)
    {
        level_str[level*2] = 0;
        for(int i = level*2-1; i >= 0; --i)
        {
            level_str[i] = ' ';
        }
    }
    else 
    {
        level_str[0] = 0;
    }

    const auto print_profiler_name_if_necessary = [logger, profiler, &printed_name, level_str]() 
    {
        if(!printed_name)
        {
            printed_name = true;
            logger.info("%s%s", level_str, profiler->_name);
        }
    };

  
    if(stats.time_us > 0)
    {
        print_profiler_name_if_necessary();
        logger.info("%s  %9s", level_str, format_microseconds_to_milliseconds(stats.time_us));
    }

    if(stats.cpu_cycles != 0)
    {
        print_profiler_name_if_necessary();
        logger.info("%s %8s CPU cycles%s",
                level_str,
                format_units(stats.cpu_cycles, 2),
                flags.isSet(Flag::ReportsFreeRunningCpuCycles) ? " (free running)" :
                flags.isSet(Flag::ExcludeFromTotalChildrenCyclesReport) ? " (excluded from child sum)" : "");
    }

    if(flags.isSet(Flag::ReportTotalChildrenCycles))
    {
        const auto children_cpu_cycles = calculate_total_children_cpu_cycles(profiler);
        if(children_cpu_cycles > 0)
        {
            print_profiler_name_if_necessary();
            logger.info("%s %8s CPU cycles (sum of all child profilers)", level_str, format_units(children_cpu_cycles, 2));
        }

        const auto children_accelerator_cycles = calculate_total_children_accelerator_cycles(profiler);
        if(children_accelerator_cycles > 0)
        {
            print_profiler_name_if_necessary();
            logger.info("%s %8s Accelerator cycles (sum of all child profilers)", level_str, format_units(children_accelerator_cycles, 2));
        }
    }

    if(!flags.isSet(Flag::ExcludeStatsFromReport))
    {
        if(stats.accelerator_cycles != 0)
        {
            print_profiler_name_if_necessary();
            logger.info("%s %8s Accelerator cycles", level_str, format_units(stats.accelerator_cycles, 2));
        }

        if(stats.time_us > 0)
        {
            if(metrics.ops != 0)
            {
                print_profiler_name_if_necessary();
                logger.info("%s %8s OP/s", level_str, format_rate(metrics.ops, stats.time_us));
            }

            if(metrics.macs != 0)
            {
                print_profiler_name_if_necessary();
                logger.info("%s %8s MAC/s", level_str, format_rate(metrics.macs, stats.time_us));
            }
        }

        for(auto item_it = profiler->custom_stats.items(); item_it != profiler->custom_stats.enditems(); ++item_it)
        {
            print_profiler_name_if_necessary();
            const auto item = item_it.current;
            logger.info("%s %s=%d", level_str, item->key, *item->value);
        }
    }

    if(profiler->_custom_stats_printer != nullptr)
    {
        print_profiler_name_if_necessary();
        profiler->_custom_stats_printer(const_cast<Profiler&>(*profiler), logger, level_str);
    }


    profiler->_was_visited = true;

    for(auto child : profiler->_children)
    {
        if(child->_was_visited)
        {
            continue;
        }

        print_profiler_statistics(child, logger, level+1, level_str, pretty_print);
    }
}

/*************************************************************************************************/
static void print_profiler_metric(
    Profiler *profiler, 
    logging::Logger &logger, 
    int level, 
    char *level_str, 
    bool pretty_print
)
{
    const auto metrics = profiler->metrics_including_children();

    if(pretty_print)
    {
        level_str[level*2] = 0;
        for(int i = level*2-1; i >= 0; --i)
        {
            level_str[i] = ' ';
        }
    }
    else 
    {
        level_str[0] = 0;
    }

    if(metrics.ops != 0 || metrics.macs != 0)
    {
        logger.info("%s%s", level_str, profiler->name());
        if(metrics.ops != 0)
        {
            logger.info("%s %7s OPs", level_str, format_units(metrics.ops, 2));
        }
        if(metrics.macs != 0)
        {
            logger.info("%s %7s MACs", level_str, format_units(metrics.macs, 2));
        }
    }
    profiler->_was_visited = true;

    for(auto child : profiler->_children)
    {
        if(child->_was_visited)
        {
            continue;
        }

        print_profiler_metric(child, logger, level+1, level_str, pretty_print);
    }
}


/*************************************************************************************************/
static logging::Logger& get_logger(logging::Logger* logger)
{
    if(logger == nullptr)
    {
        logger = logging::get("MltkProfiler");
        if(logger == nullptr)
        {
            logger = logging::create("MtlkProfiler");
        }
    }

    return *logger;
}


} // namespace profiling
