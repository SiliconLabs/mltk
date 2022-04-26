#pragma once

#include <cstdint>
#include <type_traits>

#include "tflite_model_parameters/schema/dictionary_generated.h"


namespace mltk
{


/**
 * @brief .tflite Model Parameters
 * 
 * This provides access to application parameters stored in a .tflite model flatbuffer's metadata.
 * Application parameters are things like audio sampling rate, FFT bins, etc.
 * 
 */
class TfliteModelParameters
{
private:
    typedef const flatbuffers::Vector<flatbuffers::Offset<schema::Entry>> EntryVector;
    typedef flatbuffers::Vector<flatbuffers::Offset<schema::Entry>>::const_iterator EntryVectorIterator;
    typedef flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> StringVector;
    typedef flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>::const_iterator StringVectorIterator;
    typedef flatbuffers::Vector<int32_t> Int32Vector;
    typedef flatbuffers::Vector<int32_t>::const_iterator Int32VectorIterator;
    typedef flatbuffers::Vector<float> FloatVector;
    typedef flatbuffers::Vector<float>::const_iterator FloatVectorIterator;

public:

    static const char METADATA_TAG[];


    struct StringList
    {
        struct Iterator
        {
            StringVectorIterator it;

            Iterator(StringVectorIterator it = StringVectorIterator());

            bool operator!=(Iterator rhs);
            const char* operator*();
            void operator++();
        };


        const StringVector *vector;

        StringList(const StringVector *vector = nullptr);

        uint32_t size() const;
        Iterator begin() const;
        Iterator end() const;
        const char* operator [](unsigned index) const;

        StringList(const StringList&);
        StringList& operator=(const StringList&);

        constexpr bool is_valid() const
        {
            return (vector != nullptr);
        }
    };

    struct Int32List
    {
        struct Iterator
        {
            Int32VectorIterator it;

            Iterator(Int32VectorIterator it = Int32VectorIterator());

            bool operator!=(Iterator rhs);
            int32_t operator*();
            void operator++();
        };


        const Int32Vector *vector;

        Int32List(const Int32Vector *vector = nullptr);

        uint32_t size() const;
        Iterator begin() const;
        Iterator end() const;
        int32_t operator [](unsigned index) const;

        Int32List(const Int32List&);
        Int32List& operator=(const Int32List&);

        constexpr bool is_valid() const
        {
            return (vector != nullptr);
        }
    };


    struct FloatList
    {
        struct Iterator
        {
            FloatVectorIterator it;

            Iterator(FloatVectorIterator it = FloatVectorIterator());

            bool operator!=(Iterator rhs);
            float operator*();
            void operator++();
        };


        const FloatVector *vector;

        FloatList(const FloatVector *vector = nullptr);

        uint32_t size() const;
        Iterator begin() const;
        Iterator end() const;
        float operator [](unsigned index) const;

        FloatList(const FloatList&);
        FloatList& operator=(const FloatList&);

        constexpr bool is_valid() const
        {
            return (vector != nullptr);
        }
    };

    struct Value : private schema::Entry
    {
        schema::Value type() const;
        const char* key() const;
        bool boolean() const;
        int8_t i8() const;
        uint8_t u8() const;
        int16_t i16() const;
        uint16_t u16() const;
        int32_t i32() const;
        uint32_t u32() const;
        int64_t i64() const;
        uint64_t u64() const;
        float f32() const;
        double f64() const;
        const char* str() const;
        StringList str_list() const;
        Int32List int32_list() const;
        FloatList float_list() const;
        const uint8_t* bin() const;
        uint32_t bin_length() const;
    };

    struct Iterator
    {
        EntryVectorIterator it;

        Iterator(EntryVectorIterator it);

        bool operator!=(Iterator rhs);
        const Value* operator*();
        void operator++();
    };


    static bool load_from_tflite_flatbuffer(const void* flatbuffer, TfliteModelParameters& parameters);
    bool load(const schema::Dictionary *fb_dictionary);
    bool load(const void *flatbuffer);
    void unload();

    bool contains(const char* key) const;
    const Value* operator [](const char* key) const;

    Iterator begin() const;
    Iterator end() const;

    constexpr unsigned size() const
    {
        return (_fb_dictionary == nullptr) ? 0 : _fb_dictionary->entries()->size();
    }

    constexpr bool is_loaded() const 
    {
        return _fb_dictionary != nullptr;
    }

    constexpr const schema::Dictionary* fb_dictionary() const 
    {
        return _fb_dictionary;
    }

    const Value* get(const char* key) const;
    bool get(const char* key, const char* &value) const;
    bool get(const char* key, const uint8_t* &value) const;
    bool get(const char* key, StringList &value) const;
    bool get(const char* key, Int32List &value) const;
    bool get(const char* key, FloatList &value) const;
    bool get(const char* key, const uint8_t* &data, uint32_t &length) const;
    template<typename T>
    bool get(const char* key, T &value) const
    {
        auto entry = get(key);
        if(entry != nullptr)
        {
            const auto entry_type = entry->type();
            #define _MODEL_PARAMS_ADD_GETTER(_type) \
            if(entry_type == schema::Value::_type) \
            { \
                value = static_cast<T>(entry->_type()); \
                return true; \
            }
            _MODEL_PARAMS_ADD_GETTER(boolean)
            else _MODEL_PARAMS_ADD_GETTER(i8)
            else _MODEL_PARAMS_ADD_GETTER(u8)
            else _MODEL_PARAMS_ADD_GETTER(i16)
            else _MODEL_PARAMS_ADD_GETTER(u16)
            else _MODEL_PARAMS_ADD_GETTER(i32)
            else _MODEL_PARAMS_ADD_GETTER(u32)
            else _MODEL_PARAMS_ADD_GETTER(i64)
            else _MODEL_PARAMS_ADD_GETTER(u64)
            else _MODEL_PARAMS_ADD_GETTER(f32)
            else _MODEL_PARAMS_ADD_GETTER(f64)
            #undef _MODEL_PARAMS_ADD_GETTER
        }
        return false;
    }

    TfliteModelParameters() = default;
    TfliteModelParameters(const TfliteModelParameters&);
    TfliteModelParameters& operator=(const TfliteModelParameters&);

private:
    const schema::Dictionary *_fb_dictionary = nullptr;
};


using StringList = TfliteModelParameters::StringList;
using Int32List = TfliteModelParameters::Int32List;
using FloatList = TfliteModelParameters::FloatList;

} // namespace mltk
