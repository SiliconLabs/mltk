#pragma once

#include "cpputils/helpers.hpp"
#include "cpputils/typed_list.hpp"
#include "cpputils/typed_linked_list.hpp"
#include "logging/logger.hpp"
#include "profiling/profiler_fullname.hpp"


namespace profiling
{

class Profiler;
class AveragedProfiler;
using ProfilerList = cpputils::TypedLinkedList<Profiler>;

void set_profiler_list(ProfilerList* list);
DLL_EXPORT ProfilerList& get_profilers();

bool register_profiler(const char* name, Profiler* &profiler, Profiler* parent=nullptr);
bool register_profiler(const char* name, AveragedProfiler* &profiler, Profiler* parent=nullptr);
bool unregister(const char* name, bool unregister_children=true);
bool unregister(const Profiler* profiler, bool unregister_children=true);

bool reset(const char* name, bool reset_children=true);
bool reset(Profiler* profiler, bool reset_children=true);


bool exists(const char* name);
Profiler* get(const char* name);

void get_all(const char* name, cpputils::TypedList<Profiler*>& list);

void print_metrics(const char* name, logging::Logger *logger = nullptr, bool pretty_print = true);
void print_metrics(const Profiler* profiler, logging::Logger *logger = nullptr, bool pretty_print = true);
void print_stats(const char* name, logging::Logger *logger = nullptr, bool pretty_print = true);
void print_stats(const Profiler* profiler, logging::Logger *logger = nullptr, bool pretty_print = true);
uint32_t calculate_total_children_cpu_cycles(const Profiler* profiler, bool is_root = true);
uint32_t calculate_total_children_accelerator_cycles(const Profiler* profiler, bool is_root = true);

} // namespace profiling
