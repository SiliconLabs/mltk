
#include <cassert>
#include <cstring>
#include <cstdlib>

#include "profiling/profiler.hpp"


namespace profiling
{



/*************************************************************************************************/
Profiler::Profiler(void* object_buffer, const char* name)
{
    this->_object_buffer = object_buffer;
    this->_name = name;
}

/*************************************************************************************************/
void Profiler::next(cpputils::LinkedListItem* next) 
{
    _linked_list_next = next;
}

/*************************************************************************************************/
cpputils::LinkedListItem* Profiler::next()
{
    return _linked_list_next;
}

/*************************************************************************************************/
void Profiler::unlink()
{
    assert(this->_object_buffer != nullptr);
    reset();
    _children.clear();
    custom_stats.clear();

    void *ptr = this->_object_buffer;
    this->_object_buffer = nullptr;
    free(ptr);
}

/*************************************************************************************************/
bool Profiler::parent(Profiler *parent)
{
    if(parent == nullptr)
    {
        return false;
    }
    else if(parent == this)
    {
        assert(!"Profiler cannot be a parent of itself");
        return false;
    }
    else if(parent->_children.contains(this))
    {
        assert(!"Parent already contains profiler as child");
        return false;
    }
    else if(_children.contains(parent))
    {
        assert(!"Parent is a child of this profiler");
        return false;
    }
    else
    {
        parent->_children.append(this);
        _parent = parent;
        return true;
    }
}

/*************************************************************************************************/
void Profiler::custom_stats_printer(CustomStatsPrinter* printer)
{
    _custom_stats_printer = printer;
}

/*************************************************************************************************/
int32_t Profiler::increment_custom_stat(const char* name, int32_t amount)
{
    if(!custom_stats.contains(name))
    {
        int32_t initial_value = 0;
        custom_stats.put(name, &initial_value);
    }

    int32_t* new_amount = custom_stats.get(name);
    *new_amount += amount;
    return *new_amount;
}

/*************************************************************************************************/
int32_t Profiler::get_custom_stat(const char* name) const
{
    if(!custom_stats.contains(name))
    {
        return 0;
    }
    else 
    {
        return *custom_stats.get(name);
    }
}

/*************************************************************************************************/
const char* Profiler::name() const 
{
    return _name;
}

/*************************************************************************************************/
const char* Profiler::fullname(Fullname& fullname) const
{
    Fullname::create(fullname, _name, _parent);
    return fullname.value;
}

/*************************************************************************************************/
void Profiler::reset(void)
{
    _state = State::Stopped;
    _cpu_accumulator.reset();
    _time_accumulator.reset();
    _stats.reset();
    if(_msg != nullptr)
    {
        free((void*)_msg);
        _msg = nullptr;
    }
    for(auto e : custom_stats)
    {
        *e = 0;
    }
}

/*************************************************************************************************/
void Profiler::flags(Flags flags)
{
    _flags = flags;
}

/*************************************************************************************************/
void Profiler::msg(const char* msg)
{
    if(_msg != nullptr)
    {
        free((void*)_msg);
        _msg = nullptr;
    }

    auto m = static_cast<char*>(malloc(strlen(msg) + 1));
    if(m != nullptr)
    {
        strcpy(m, msg);
        _msg = m;
    }
}

/*************************************************************************************************/
const char* Profiler::msg() const
{
    return (_msg == nullptr) ? "" : _msg;
}

/*************************************************************************************************/
Metrics Profiler::metrics_including_children() const
{
    Metrics metrics;

    get_child_metrics(this, metrics);

    return metrics;
}

/*************************************************************************************************/
void Profiler::get_child_metrics(const Profiler *profiler, Metrics &metrics) const
{
    metrics += profiler->_metrics;

    for(auto child : profiler->_children)
    {
        get_child_metrics(child, metrics);
    }
}

/*************************************************************************************************
 * Update this profiler's stats after stopping or pausing
 */
void Profiler::update_stats(bool stop, uint32_t stop_cpu_cycle, uint32_t stop_time_us)
{
    // If the profiler was already stopped
    // then just return
    if(_state == State::Stopped)
    {
        return;
    }

    // If the profiler was started
    if(_state == State::Started)
    {
        // Then update the accumulation now that we're
        // pausing or stopping
        const uint32_t cpu_diff = stop_cpu_cycle - _cpu_accumulator.start_marker;
        _cpu_accumulator.accumulator += cpu_diff;
        const uint32_t time_diff = stop_time_us - _time_accumulator.start_marker;
        _time_accumulator.accumulator += time_diff;
    }

    // If we're stopping the profiler
    if(stop)
    {
        _state = State::Stopped;
#ifdef __arm__
        _stats.cpu_cycles = _cpu_accumulator.accumulator;

        // Either the time is measure between the start and stop of the profiler
        // OR the time is measured as the accumlation between start/pause and start/stop
        if(_flags.isSet(Flag::TimeMeasuredBetweenStartAndStop))
        {
            _stats.time_us = stop_time_us - _time_accumulator.start_base;
        }
        else 
        {
            _stats.time_us = _time_accumulator.accumulator;
        }
#endif // __arm__
        
        _cpu_accumulator.reset();
        _time_accumulator.reset();
    }
    else
    {
        _state = State::Paused;
    }
}

/*************************************************************************************************/
void AveragedProfiler::update_stats(bool stop, uint32_t stop_cpu_cycle, uint32_t stop_time_us)
{
    if(_state != State::Stopped)
    {
        if(_state == State::Started)
        {
            const uint32_t cpu_diff = stop_cpu_cycle - _cpu_accumulator.start_marker;
            _cpu_accumulator.accumulator += cpu_diff;
            const uint32_t time_diff = stop_time_us - _time_accumulator.start_marker;
            _time_accumulator.accumulator += time_diff;
        }

        if(stop)
        {
            _state = State::Stopped;
            ++total_count;
            total_cpu_cycles += _cpu_accumulator.accumulator;
            total_time_us += _time_accumulator.accumulator;
            _stats.cpu_cycles = total_cpu_cycles / total_count;
            _stats.time_us = total_time_us / total_count;
            _cpu_accumulator.reset();
            _time_accumulator.reset();
        }
        else
        {
            _state = State::Paused;
        }
    }
}

/*************************************************************************************************/
void AveragedProfiler::reset(void)
{
    Profiler::reset();
    total_count = 0;
    total_cpu_cycles = 0;
    total_time_us = 0;
}


} // namespace profiling
