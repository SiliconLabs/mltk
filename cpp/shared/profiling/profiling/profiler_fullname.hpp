


namespace profiling
{

class Profiler;

class Fullname
{
public:
    static const char SEPARATOR[];

    char value[128] = "";
    
    Fullname() = default;
    
    bool is_valid() const 
    {
        return !_invalid;
    }

    static bool create(Fullname& fullname, const char* name, Profiler* parent = nullptr);

protected:
    char *ptr = nullptr;
    bool _invalid = false;

    void append(const char* name);
};


} // namespace profiling
