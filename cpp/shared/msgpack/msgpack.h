#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


#define _MSGPACK_GET_OBJ_VALUE_(obj, obj_type, name) ((const obj_type*)(obj))->data.name
#define _MSGPACK_IS_(_obj, _type)   ((_obj != NULL) && (((const msgpack_object_t*)_obj)->type == MSGPACK_TYPE_ ## _type))

/**
 * Get MessagePack object as boolean
 */
#define MSGPACK_BOOL(obj)               _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object8_t, boolean)
/**
 * Get MessagePack object as signed 8bit
 * @see @ref MSGPACK_INT() to automatically determine the integer bit size
 */
#define MSGPACK_INT8(obj)               _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object8_t, s)
/**
 * Get MessagePack object as unsigned 8bit
 * @see @ref MSGPACK_UINT() to automatically determine the integer bit size
 */
#define MSGPACK_UINT8(obj)              _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object8_t, u)
/**
 * Get MessagePack object as signed 16bit
 * @see @ref MSGPACK_INT() to automatically determine the integer bit size
 */
#define MSGPACK_INT16(obj)              _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object16_t, s)
/**
 * Get MessagePack object as unsigned 16bit
 * @see @ref MSGPACK_UINT() to automatically determine the integer bit size
 */
#define MSGPACK_UINT16(obj)             _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object16_t, u)
/**
 * Get MessagePack object as signed 32bit
 * @see @ref MSGPACK_INT() to automatically determine the integer bit size
 */
#define MSGPACK_INT32(obj)              _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object32_t, s)
/**
 * Get MessagePack object as unsigned 32bit
 * @see @ref MSGPACK_UINT() to automatically determine the integer bit size
 */
#define MSGPACK_UINT32(obj)             _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object32_t, u)
/**
 * Get MessagePack object as float
 */
#define MSGPACK_FLOAT(obj)              _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object32_t, flt)
/**
 * Get MessagePack object as signed 64bit
 * @see @ref MSGPACK_LONG() to automatically determine the integer bit size
 */
#define MSGPACK_INT64(obj)              _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object64_t, s)
/**
 * Get MessagePack object as unsigned 64bit
 * @see @ref MSGPACK_ULONG() to automatically determine the integer bit size
 */
#define MSGPACK_UINT64(obj)             _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object64_t, u)
/**
 * Get MessagePack object as double
 */
#define MSGPACK_DOUBLE(obj)             _MSGPACK_GET_OBJ_VALUE_(obj, msgpack_object64_t, dbl)
/**
 * Get MessagePack object with specific key in dictionary (aka map)
 */
#define MSGPACK_DICT(obj, key)          msgpack_get_dict_object((msgpack_object_dict_t*)obj, key, MSGPACK_TYPE_ANY)
/**
 * Return the number of entries in a dict
 */
#define MSGPACK_DICT_LENGTH(obj)       (((msgpack_object_dict_t*)obj)->count)
/**
 * Get MessagePack dictionary entry with a specific type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_DICT_TYPE(obj, key, type) msgpack_get_dict_object((msgpack_object_dict_t*)obj, key, MSGPACK_TYPE_ ## type)
/**
 * Get MessagePack dictionary entry with a STR type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_DICT_STR(obj, key)      (msgpack_object_str_t*)MSGPACK_DICT_TYPE(obj, key, STR)
/**
 * Get MessagePack dictionary entry with a BIN type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_DICT_BIN(obj, key)      (msgpack_object_bin_t*)MSGPACK_DICT_TYPE(obj, key, BIN)
/**
 * Get MessagePack dictionary entry with a DICT type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_DICT_DICT(obj, key)     (msgpack_object_dict_t*)MSGPACK_DICT_TYPE(obj, key, DICT)
/**
 * Get MessagePack dictionary entry with a ARRAY type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_DICT_ARRAY(obj, key)    (msgpack_object_array_t*)MSGPACK_DICT_TYPE(obj, key, ARRAY)
/**
 * Get MessagePack dictionary entry with an signed integer (unsigned and bool included) type.
 * Use MSGPACK_DICT_TYPE(obj, key, INT) if you specifically need a signed integer.
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_DICT_INT(obj, key)      MSGPACK_DICT_TYPE(obj, key, INT_OR_UINT)
/**
 * Get MessagePack dictionary entry with a unsigned integer (signed and bool included) type,
 * Use MSGPACK_DICT_TYPE(obj, key, UINT) if you specifically need an unsigned integer.
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_DICT_UINT(obj, key)     MSGPACK_DICT_TYPE(obj, key, INT_OR_UINT)

/**
 * Get MessagePack object at specific index of array
 */
#define MSGPACK_ARRAY(obj, index)       msgpack_get_array_object((msgpack_object_array_t*)obj, index, MSGPACK_TYPE_ANY)
/**
 * Get MessagePack array at index with a specific type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_ARRAY_TYPE(obj, index, type) msgpack_get_array_object((msgpack_object_array_t*)obj, index, MSGPACK_TYPE_ ## type)
/**
 * Get MessagePack array at index with a DICT type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_ARRAY_DICT(obj, index)  (msgpack_object_dict_t*)MSGPACK_ARRAY_TYPE(obj, index, DICT)
/**
 * Get MessagePack array at index with a ARRAY type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_ARRAY_ARRAY(obj, index) (msgpack_object_array_t*)MSGPACK_ARRAY_TYPE(obj, index, ARRAY)
/**
 * Return the number of entries in an array
 */
#define MSGPACK_ARRAY_LENGTH(obj)       (((msgpack_object_array_t*)obj)->count)
/**
 * Get MessagePack array at index with a STR type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_ARRAY_STR(obj, index)   (msgpack_object_str_t*)MSGPACK_ARRAY_TYPE(obj, index, STR)
/**
 * Get MessagePack array at index with a BIN type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_ARRAY_BIN(obj, index)   (msgpack_object_bin_t*)MSGPACK_ARRAY_TYPE(obj, index, BIN)
/**
 * Get MessagePack array at index with a signed integer (unsigned and bool included) type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_ARRAY_INT(obj, index)   MSGPACK_ARRAY_TYPE(obj, index, INT)
/**
 * Get MessagePack array at index with a unsigned integer (signed and bool included) type,
 * return NULL if the entry is found but the wrong type
 */
#define MSGPACK_ARRAY_UINT(obj, index)  MSGPACK_ARRAY_TYPE(obj, index, UINT)

/**
 * Get string MessagePack object's string value
 * @note The string value is NOT null-terminated
 * @see @ref MSGPACK_STR()
 */
#define MSGPACK_STR_VALUE(obj)          ((const msgpack_object_str_t*)(obj))->data
/**
 * Get MessagePack object string length
 */
#define MSGPACK_STR_LENGTH(obj)         ((const msgpack_object_str_t*)(obj))->length
/**
 * Compare string MessagePack object to another string
 */
#define MSGPACK_STR_CMP(obj, str)       msgpack_str_cmp((msgpack_object_t*)obj, str)
/**
 * Get binary MessagePack object's value
 */
#define MSGPACK_BIN_VALUE(obj)          ((const msgpack_object_bin_t*)(obj))->data
/**
 * Get binary MessagePack object's length
 */
#define MSGPACK_BIN_LENGTH(obj)         ((const msgpack_object_bin_t*)(obj))->length
/**
 * Compare binary MessagePack object to another binary value
 */
#define MSGPACK_BIN_CMP(obj, mem)       memcmp(MSGPACK_BIN_VALUE(obj), mem, MSGPACK_BIN_LENGTH(obj))

/**
 * Get MessagePack object as a signed 32bit value
 * @see msgpack_get_int() for more info
 */
#define MSGPACK_INT(obj)                msgpack_get_int(obj)
/**
 * Get MessagePack object as an unsigned 32bit value
 * @see msgpack_get_int() for more info
 */
#define MSGPACK_UINT(obj)               msgpack_get_uint(obj)
/**
 * Get MessagePack object as a signed 64bit value
 * @see msgpack_get_int() for more info
 */
#define MSGPACK_LONG(obj, buffer)       msgpack_get_long(obj, buffer)
/**
 * Get MessagePack object as an unsigned 64bit value
 * @see msgpack_get_int() for more info
 */
#define MSGPACK_ULONG(obj, buffer)      msgpack_get_ulong(obj, buffer)
/**
 * Copy STR obj contents to buffer with null-terminator
 * @see msgpack_get_str() for more info
 */
#define MSGPACK_STR(obj, buffer, max_length)  msgpack_get_str((msgpack_object_t*)obj, buffer, max_length)
/**
 * Copy BIN obj contents to buffer
 */
#define MSGPACK_BIN(obj, mem, max_length) memcpy(mem, MSGPACK_BIN_VALUE(obj), MIN(max_length, MSGPACK_BIN_LENGTH(obj)))

/**
 * Convert an object to a string and add to buffer
 * @note The buffer must be at least 32 bytes
 */
#define MSGPACK_TO_STR(obj, buffer, max_length) msgpack_to_str((msgpack_object_t*)obj, buffer, max_length)

/**
 * Return true if MessagePack object is a NIL (i.e. NULL value)
 */
#define MSGPACK_IS_NIL(obj)         _MSGPACK_IS_(obj, NIL)
/**
 * Return true if MessagePack object is a boolean
 */
#define MSGPACK_IS_BOOL(obj)        _MSGPACK_IS_(obj, BOOL)
/**
 * Return true if MessagePack object is a signed 8bit integer
 */
#define MSGPACK_IS_INT8(obj)        _MSGPACK_IS_(obj, INT8)
/**
 * Return true if MessagePack object is an unsigned 8bit integer
 */
#define MSGPACK_IS_UINT8(obj)       _MSGPACK_IS_(obj, UINT8)
/**
 * Return true if MessagePack object is a signed 16bit integer
 */
#define MSGPACK_IS_INT16(obj)       _MSGPACK_IS_(obj, INT16)
/**
 * Return true if MessagePack object is an unsigned 16bit integer
 */
#define MSGPACK_IS_UINT16(obj)      _MSGPACK_IS_(obj, UINT16)
/**
 * Return true if MessagePack object is a signed 32bit integer
 */
#define MSGPACK_IS_INT32(obj)       _MSGPACK_IS_(obj, INT32)
/**
 * Return true if MessagePack object is an unsigned 32bit integer
 */
#define MSGPACK_IS_UINT32(obj)      _MSGPACK_IS_(obj, UINT32)
/**
 * Return true if MessagePack object is a signed 64bit integer
 */
#define MSGPACK_IS_INT64(obj)       _MSGPACK_IS_(obj, INT64)
/**
 * Return true if MessagePack object is an unsigned 64bit integer
 */
#define MSGPACK_IS_UINT64(obj)      _MSGPACK_IS_(obj, UINT64)
/**
 * Return true if MessagePack object is an unsigned integer
 */
#define MSGPACK_IS_UINT(obj)        msgpack_object_is_type((msgpack_object_t*)obj, MSGPACK_TYPE_UINT)
/**
 * Return true if MessagePack object is an signed integer
 */
#define MSGPACK_IS_INT(obj)        msgpack_object_is_type((msgpack_object_t*)obj, MSGPACK_TYPE_INT)
/**
 * Return true if MessagePack object is a float
 */
#define MSGPACK_IS_FLOAT(obj)       _MSGPACK_IS_(obj, FLOAT)
/**
 * Return true if MessagePack object is a double
 */
#define MSGPACK_IS_DOUBLE(obj)      _MSGPACK_IS_(obj, DOUBLE)
/**
 * Return true if MessagePack object is a string
 */
#define MSGPACK_IS_STR(obj)         _MSGPACK_IS_(obj, STR)
/**
 * Return true if MessagePack object is a binary string
 */
#define MSGPACK_IS_BIN(obj)         _MSGPACK_IS_(obj, BIN)
/**
 * Return true if MessagePack object is a dictionary (aka map)
 */
#define MSGPACK_IS_DICT(obj)        _MSGPACK_IS_(obj, DICT)
/**
 * Return true if MessagePack object is an array
 */
#define MSGPACK_IS_ARRAY(obj)       _MSGPACK_IS_(obj, ARRAY)

/**
 * Raw value of a MessagePack NIL typed object
 */
#define MSGPACK_NIL     0xC0
/**
 * Raw value of a MessagePack FALSE boolean typed object
 */
#define MSGPACK_FALSE   0xC2
/**
 * Raw value of a MessagePack TRUE boolean typed object
 */
#define MSGPACK_TRUE    0xC3


/**
 * Return the number of data buffer bytes currently used by a @ref msgpack_context_t
 *
 * @param ctx @ref msgpack_context_t initialized with a data buffer
 */
#define MSGPACK_BUFFER_USED(ctx) (uintptr_t)((ctx)->buffer.ptr - (ctx)->buffer.buffer)
/**
 * Return the number of data buffer bytes unused (i.e. available) by a @ref msgpack_context_t
 * @param ctx @ref msgpack_context_t initialized with a data buffer
 */
#define MSGPACK_BUFFER_REMAINING(ctx) (uintptr_t)((ctx)->buffer.end - (ctx)->buffer.ptr)

/**
 * Return an object for the corresponding key in a dictionary
 * @see msgpack_get_dict_object for more info
 */
#define MSGPACK_DICT_GET_OBJECT(dict, key) msgpack_get_dict_object((const msgpack_object_dict_t*)dict, key, MSGPACK_TYPE_ANY)

/**
 * Return the object a the corresponding index in an array
 * @see msgpack_get_array_object() for more info
 */
#define MSGPACK_ARRAY_GET_OBJECT(array, index) msgpack_get_array_object((msgpack_object_array_t*)array, index, MSGPACK_TYPE_ANY)

/**
 * Non-recursively iterate the key/value pairs of a dictionary or entries in array
 * @see msgpack_foreach() for more info
 */
#define MSGPACK_FOREACH(dict_or_array, callback, arg) msgpack_foreach((const msgpack_object_t*)dict_or_array, callback, arg, 0)

/**
 * Recursively iterate the key/value pairs of a dictionary or entries in array
 * @see msgpack_foreach() for more info
 */
#define MSGPACK_FOREACH_RECURSIVE(dict_or_array, callback, arg) msgpack_foreach((const msgpack_object_t*)dict_or_array, callback, arg, UINT32_MAX)


/**
 * De-allocate all objects assoicated with supplied object
 * @see msgpack_free_objects() for more info
 */
#define MSGPACK_FREE_OBJECTS(obj) msgpack_free_objects(obj)




#pragma pack(push, 1)

/**
 * MessagePack object data type
 */
typedef uint8_t msgpack_type_t;
enum msgpack_type_enum
{
    MSGPACK_TYPE_NIL,       //!< Null value
    MSGPACK_TYPE_BOOL,      //!< Boolean
    MSGPACK_TYPE_INT8,      //!< Signed, 8bit integer
    MSGPACK_TYPE_UINT8,     //!< Unsigned, 8bit integer
    MSGPACK_TYPE_INT16,     //!< Signed, 16bit integer
    MSGPACK_TYPE_UINT16,    //!< Unsigned, 16bit integer
    MSGPACK_TYPE_INT32,     //!< Signed, 32bit integer
    MSGPACK_TYPE_UINT32,    //!< Unsigned, 32bit integer
    MSGPACK_TYPE_FLOAT,     //!< Single precision, floating point number
    MSGPACK_TYPE_INT64,     //!< Signed, 64bit integer
    MSGPACK_TYPE_UINT64,    //!< Unisgned, 64but integer
    MSGPACK_TYPE_DOUBLE,    //!< Double precision, floating point number
    MSGPACK_TYPE_STR,       //!< ASCII string
    MSGPACK_TYPE_BIN,       //!< Binary string
    MSGPACK_TYPE_DICT,      //!< Dictionary (a.k.a. map)
    MSGPACK_TYPE_ARRAY,     //!< Array

    // The following are non-standard types, used internally by this library
    MSGPACK_TYPE_ANY,       //!< Any data type
    MSGPACK_TYPE_INT,       //!< Any signed integer
    MSGPACK_TYPE_UINT,       //!< Any unsigned integer
    MSGPACK_TYPE_INT_OR_UINT //!< Any signed or unsigned integer
};



/**
 * De-serialized @ref msgpack_object_t flags
 */
typedef uint8_t msgpack_object_flag_t;
enum msgpack_object_flag_enum
{
    MSGPACK_OBJECT_FLAG_NONE                = 0,            //!< No flags
    MSGPACK_OBJECT_FLAG_WAS_ALLOCATED       = (1 << 0),     //!< The object as allocated from the heap
    MSGPACK_OBJECT_FLAG_AUTO_FREE_USER      = (1 << 1)      //!< When the object is freed with @ref msgpack_free_objects(),
                                                            //!< also free the `user context` set with @ref msgpack_set_user_context() (if applicable)
};

/**
 * Generic MessagePack object
 */
typedef struct
{
    msgpack_type_t type;        ///< The object's type, see @ref msgpack_type_t
    msgpack_object_flag_t flags;///< Object flags, see @ref msgpack_object_flag_t
    uint8_t reserved[2];
} msgpack_object_t;

/**
 * 8-bit data
 */
typedef union
{
    bool    boolean;
    uint8_t         u;
    int8_t          s;
} msgpack_data8_t;

/**
 * 8-bit data object
 */
typedef struct
{
    msgpack_object_t obj;
    msgpack_data8_t data;
} msgpack_object8_t;

/**
 * 16-bit data
 */
typedef  union
{
    uint16_t  u;
    int16_t   s;
} msgpack_data16_t;

/**
 * 16-bit data object
 */
typedef struct
{
    msgpack_object_t obj;
    msgpack_data16_t data;
} msgpack_object16_t;

/**
 * 32-bit data
 */
typedef  union
{
    uint32_t  u;
    int32_t   s;
    float     flt;
} msgpack_data32_t;

/**
 * 32-bit data object
 */
typedef struct
{
    msgpack_object_t obj;
    msgpack_data32_t data;
} msgpack_object32_t;

/**
 * 64-bit data
 */
typedef  union
{
    uint64_t  u;
    int64_t   s;
    double    dbl;
} msgpack_data64_t;

/**
 * 64-bit data object
 */
typedef struct
{
    msgpack_object_t obj;
    msgpack_data64_t data;
} msgpack_object64_t;

/**
 * MessagePack string object
 */
typedef struct
{
    msgpack_object_t obj;
    uint32_t length;
    char *data;
} msgpack_object_str_t;

/**
 * MessagePack binary string object
 */
typedef struct
{
    msgpack_object_t obj;
    uint32_t length;
    void *data;
} msgpack_object_bin_t;

/**
 * MessagePack dictionary entry
 */
typedef struct
{
    msgpack_object_t *key;
    msgpack_object_t *value;
} msgpack_dict_entry_t;

/**
 * MessagePack Dictionary (aka map) object
 *
 * A dictionary is composed of zero or more key/value pairs
 *
 * @note The key is already a string object.
 *
 * @see @ref msgpack_get_dict_object() for retrieving objects from a dictionary
 */
typedef struct
{
    msgpack_object_t obj;
    uint32_t count;
    msgpack_dict_entry_t entries[];
}  msgpack_object_dict_t;

/**
 * MessagePack Array object
 *
 * An array is composed of zero or more objects
 *
 * @see @ref msgpack_get_array_object() for retrieving objects from an array
 */
typedef struct
{
    msgpack_object_t obj;
    uint32_t count;
    msgpack_object_t *entries[];
} msgpack_object_array_t;


#pragma pack(pop)


/**
 * Callback for iterating a dictionary or array
 *
 * This is invoked for each key/value pair of a @ref msgpack_object_dict_t
 * or each entry of a @ref msgpack_object_array_t supplied to @ref msgpack_foreach().
 *
 * This callback is invoked in the context of the caller of @ref msgpack_foreach().
 * If this callback returns:
 * - 0 - then continue to the next key/value pair or entry
 * - else - abort the iteration and return the given result to the caller of @ref msgpack_foreach().
 *
 * @param[in] key Dictionary key if iterating a directory, NULL if iterating an array
 * @param[in] value Dictionary entry value or array entry
 * @param[in] arg Custom argument passed to @ref msgpack_foreach()
 * @return @ref 0 to keep iterating, else abort iteration and return result to caller
 */
typedef int (*msgpack_iterator_t)(const msgpack_object_t *key, const msgpack_object_t *value, void *arg);


/**
 * MessagePack data writer
 *
 * When using the msgpack_write_xxx APIs this callback will be invoked to write
 * the 'packed' message data IF the @ref msgpack_context_t is initialized with a writer
 *
 * @param user Optional user object specified when initializing the @ref msgpack_context_t context
 * @param data 'packed' message data to write
 * @param length Length of data to write
 * @return 0 if success, else failure
 */
typedef int (*msgpack_writer_t)(void *user, const void *data, uint32_t length);


/**
 * @brief Deserialization flags
 */
typedef uint32_t msgpack_flag_t;
enum msgpack_flag_enum
{
    MSGPACK_FLAGS_NONE = 0,                                 //!< No flags
    MSGPACK_DESERIALIZE_WITH_PERSISTENT_STRINGS = (1 << 0), //!< If specified, strings within the de-serialize objects will persist after the provided buffer is freed
                                                            //!< If NOT specified, then strings will NOT be valid after the provided buffer is freed
    MSGPACK_PACK_16BIT_DICTS                    = (1 << 1), //!< If specified, the dictionary length is always 16bits
                                                            //!< If NOT specified, the dictionary length is variable based on the provided element count

    _MSGPACK_BUFFERED_WRITER                    = (1 << 31) //!< This is used internally to indicate if the contents was initialized as a buffered writer
};


#ifndef MSGPACK_MAX_NESTED_CONTAINERS
/**
 * If defined, this library will keep track of the element count
 * as they're written to dicts and arrays. This defines the maximum
 * amount of nesting that is supported.
 * @note This must be defined to use dynamic dicts and arrays
*/
#define MSGPACK_MAX_NESTED_CONTAINERS 16
#endif

/**
 * MessagePack data writing context
 *
 * This context should be initialized before calling the msgpack_write_xxx APIs
 *
 *
 * @note Either the `writer` OR `buffer` members should be populated, NOT both.
 *
 * @see @ref msgpack_init_with_buffer() and @ref msgpack_init_with_writer() helper macros
 */
typedef struct
{
    struct
    {
        uint8_t *buffer;        ///< buffer to hold 'packed' message data
        uint8_t *ptr;           ///< Pointer to unused address in buffer
        uint8_t *end;           ///< Address of end of buffer ( end = buffer + sizeof(buffer) )
    } buffer;
    msgpack_writer_t writer;    ///< Data writer, see @ref msgpack_writer_t

    void *user;                 ///< Optional user argument to pass to @ref msgpack_writer_t
#ifdef MSGPACK_MAX_NESTED_CONTAINERS
    struct
    {
        int32_t count;          ///< Either the number of elements remaining to be written to the current container
                                ///< or the number of elements that have been written to a dynamically sized container
        uint32_t marker_offset; ///< If using a dynamically sized container, this points to the buffer offset of the container's marker
    } containers[MSGPACK_MAX_NESTED_CONTAINERS];
#endif
    int32_t container_index;    ///< The current nested container (dict or array) index
    msgpack_flag_t flags;       ///< Optional flags, see @ref msgpack_flag_t
} msgpack_context_t;



/**
 * Declare and initialize a @ref msgpack_context_t with the supplied buffer
 *
 * @param buffer Data buffer to initialize context with
 * @param uint32_t Length of supplied data buffer
 * @return Initialize @ref msgpack_context_t
 */
static inline msgpack_context_t msgpack_init_with_buffer(uint8_t* buffer, uint32_t length)
{
    msgpack_context_t context =
    {
        .buffer =
        {
            .buffer = buffer,
            .ptr = buffer,
            .end = buffer + length,
        },
        .writer = NULL,
        .user = NULL,
        .container_index = -1,
        .flags = 0,
    };
    return context;
}


/**
 * Declare and initialize a @ref msgpack_context_t with the supplied @ref msgpack_writer_t
 *
 * @param writer MessagePack data writer, see @ref msgpack_writer_t
 * @param user Optional user argument given to the data writer
 */
static inline msgpack_context_t msgpack_init_with_writer(msgpack_writer_t writer, void* user)
{
    msgpack_context_t context =
    {
        .buffer =
        {
            .buffer = NULL,
            .ptr = NULL,
            .end = NULL,
        },
        .writer = writer,
        .user = user,
        .container_index = -1,
        .flags = 0,
    };
    return context;
}

/**
 * Configure the given @ref msgpack_context_t with a @ref msgpack_writer_t
 *
 * @param context Pointer to previously declared/allocated @ref msgpack_context_t
 * @param writer MessagePack data writer, see @ref msgpack_writer_t
 * @param user Optional user argument given to the data writer
 */
static inline void msgpack_set_writer(msgpack_context_t *context, msgpack_writer_t writer, void* user)
{
    context->writer = writer;
    context->user = user;
}


/**
 * Convert a 'packed' MessagePack binary string to a linked list of @ref msgpack_object_t
 *
 * This will parse the binary MessagePack data into a linked list of @ref msgpack_object_t.
 * Each @ref msgpack_object_t is dynamically allocated.
 *
 * Use @ref msgpack_free_objects() to release the allocated memory.
 *
 * By default, this function does NOT allocate additional memory for binary strings and strings.
 * Thus the given `buffer` MUST persist for as long as the return objects are referenced.
 * If the given `buffer` is released, the memory a @ref MSGPACK_TYPE_STR or @ref MSGPACK_TYPE_BIN
 * object references will be invalid.
 * Use the flag @ref MSGPACK_DESERIALIZE_WITH_PERSISTENT_STRINGS to ensure that the object data persists after the given `buffer` is de-allocated.

 * @note The returned root object will be a @ref msgpack_object_array_t OR @ref msgpack_object_dict_t
 *
 * @param root_ptr Pointer to hold allocated linked list of MessagePack object
 * @param buffer Buffer containing 'packed' MessagePack data, this buffer MUST persist for as long as the return object are referenced IF binary strings or strings are used
 * @param length Length of buffer to parse
 * @param flags @ref msgpack_flag_t
 * @return 0 on success, else failure
 */
int msgpack_deserialize_with_buffer(msgpack_object_t **root_ptr, const void *buffer, uint32_t length, msgpack_flag_t flags);

/**
 * Release all @ref msgpack_object_t the supplied object references
 *
 * This should be used to release the memory that's allocated with @ref msgpack_deserialize_with_buffer()
 * @param root_obj Object to release
 */
void msgpack_free_objects(void *root_obj);

/**
 * Set user context for object
 *
 * This associates a pointer to the given `obj`.
 * This `obj` MUST be a dictionary or array object.
 *
 * Optionally, if `auto_free=true`, when the `obj` is cleaned up, the given `user` context will also be cleaned via @ref free()
 *
 * One use-case of this is to set the user context pointer as the `buffer` argument that was given to
 * @ref msgpack_deserialize_with_buffer(). This way when the root object is cleaned the associated `buffer` is also automatically cleaned.
 *
 * @note Use @ref msgpack_get_user_context() to get the user context pointer
 *
 * @param obj Dictionary or array msgpack object
 * @param user User pointer to link to given `obj`
 * @param auto_free If true, the `user` is automatically freed via @ref free() when the `obj` is cleaned
 * @return @ref result_t of API call
 */
int msgpack_set_user_context(msgpack_object_t *obj, void *context, bool auto_free);

/**
 * Retrieve user context pointer lined to object
 *
 * This returns the user context pointer that was set via @ref msgpack_set_user_context()
 *
 * @param obj Object to retrieve user context pointer from
 * @param user_ptr Pointer to hold user context
 * @return @ref result_t of API call
 */
int msgpack_get_user_context(const msgpack_object_t *obj, void **context_ptr);

/**
 * Return the value of the corresponding key in dictionary
 *
 * This returns the value which has the given `key` in the supplied `dict`
 *
 * If the `key` is not found or the `dict` is invalid, NULL is returned.
 *
 * @param dict @ref msgpack_object_dict_t A dictionary object
 * @param key The dictionary key of the desired value to return
 * @param type The expected type of object to return, see @ref msgpack_type_t, set to @ref MSGPACK_TYPE_ANY to return any type
 * @return The dictionary value IF found, NULL else
 */
msgpack_object_t* msgpack_get_dict_object(const msgpack_object_dict_t *dict, const char* key, msgpack_type_t type);

/**
 * Return value of corresponding index in array
 *
 * This returns the value at the given `index` in the supplied `array`
 *
 * If the `index` is out-of-range or the `array` is invalid, NULL is returned.
 *
 * @param array @ref msgpack_object_array_t An array object
 * @param index Index to return value
 * @param type The expected type of object to return, see @ref msgpack_type_t, set to @ref MSGPACK_TYPE_ANY to return any type
 * @return Array value IF found, NULL else
 */
msgpack_object_t* msgpack_get_array_object(const msgpack_object_array_t *array, uint32_t index, msgpack_type_t type);

/**
 * Iterate the entries in a dictionary or array
 *
 * This invokes the supplied `iterator_callback` for each entry in the supplied `dict_or_array`.
 * See @ref msgpack_iterator_t for more details on how the callback works.
 *
 * The `arg` argument will be supplied to the `iterator_callback` callback.
 *
 * If the `depth` argument is greater than zero then dictionary/array entries will be recursively iterated.
 *
 *
 * @param dict_or_array A @ref msgpack_object_dict_t or @ref msgpack_object_array_t object to iterate
 * @param iterator_callback @ref msgpack_iterator_t callback to be invoked for each entry in the dictionary or array
 * @param arg Custom argument to pass to the `iterator_callback`
 * @param recursive_depth Recursion depth, 0 will only iterate the entries in the given `dict_or_array`, greater than 0 will iterate sub array/dictionaries
 * @return 0 if all all entries iterated, else failure
 */
int msgpack_foreach(
    const msgpack_object_t *dict_or_array,
    msgpack_iterator_t iterator_callback,
    void *arg,
    uint32_t recursive_depth
);

/**
 * Recursively dump the contents of the given dict or array as a string
 *
 * @param dict_or_array A @ref msgpack_object_dict_t or @ref msgpack_object_array_t object to iterate
 * @param recursive_depth Recursion depth, 0 will only iterate the entries in the given `dict_or_array`, greater than 0 will iterate sub array/dictionaries
 * @param write_callback Callback to invoke to write the generate strings
 * @param write_callback_arg Optional argument to pass to `write_callback`
 *
 * @return 0 if all all entries dumped, else failure
*/
int msgpack_dump(
    const msgpack_object_t *dict_or_array,
    uint32_t recursive_depth,
    void (*write_callback)(const char*, void*),
    void *write_callback_arg
);

/**
 * Convert a @ref msgpack_object_t to a string
 *
 * Convert a @ref msgpack_object_t to a string and add to provided buffer.
 *
 * @note The `max_length` MUST be at least 32 bytes
 *
 * @param obj @ref msgpack_object_t to convert to string
 * @param buffer Buffer to hold string
 * @param max_length The size of the provided buffer in bytes
 * @return Same pointer as `buffer` argument
 */
char* msgpack_to_str(const msgpack_object_t *obj, char *buffer, uint32_t max_length);

/**
 * Remove an entry from a dictionary object
 *
 * This removes and de-allocates the given object from the dictionary.
 *
 * @param dict_obj A dictionary object
 * @param obj Object to remove and de-allocate from dictionary
 * @return @ref result_t of API call
 */
int  msgpack_remove_dict_object(msgpack_object_dict_t *dict_obj, void *obj);

/**
 * Return if an object is a given type

 * @param object msgpack object, see @ref msgpack_object_t
 * @param type msgpack object type, see @ref msgpack_type_t
 */
bool msgpack_object_is_type(const msgpack_object_t *object, msgpack_type_t type);

/**
 * Return a MessagePack integer object as a signed 32bit value
 *
 * This converts the following @ref msgpack_type_t to a signed 32bit value
 * - @ref MSGPACK_TYPE_BOOL
 * - @ref MSGPACK_TYPE_INT8
 * - @ref MSGPACK_TYPE_UINT8
 * - @ref MSGPACK_TYPE_INT16
 * - @ref MSGPACK_TYPE_UINT16
 * - @ref MSGPACK_TYPE_INT32
 * - @ref MSGPACK_TYPE_UINT32
 *
 * This is useful is the actual bit-length is unknown
 *
 * @param object Integer object type to return as signed 32bit value
 * @return signed 32bit value of supplied object
 */
int32_t msgpack_get_int(const msgpack_object_t *object);

/**
 * Return a MessagePack integer object as an unsigned 32bit value
 *
 * This converts the following @ref msgpack_type_t to an unsigned 32bit value
 * - @ref MSGPACK_TYPE_BOOL
 * - @ref MSGPACK_TYPE_INT8
 * - @ref MSGPACK_TYPE_UINT8
 * - @ref MSGPACK_TYPE_INT16
 * - @ref MSGPACK_TYPE_UINT16
 * - @ref MSGPACK_TYPE_INT32
 * - @ref MSGPACK_TYPE_UINT32
 *
 * This is useful is the actual bit-length is unknown
 *
 * @param object Integer object type to return as unsigned 32bit value
 * @return unsigned 32bit value of supplied object
 */
uint32_t msgpack_get_uint(const msgpack_object_t *object);

/**
 * Return a MessagePack integer object as a signed 64bit value
 *
 * This has exact same functionality as @ref msgpack_get_int() except
 * it also supports the following @ref msgpack_type_t :
 * - @ref MSGPACK_TYPE_INT64
 * - @ref MSGPACK_TYPE_UINT64
 *
 * This is useful is the actual bit-length is unknown
 *
 * @param object Integer object to return a signed 64bit value
 * @param buffer Buffer to hold 64bit value (must be at least 8 bytes)
 * @return Pointer to 64bit value (same pointer as supplied `buffer`)
 */
int64_t* msgpack_get_long(const msgpack_object_t *object, int64_t *buffer);

/**
 * Return a MessagePack integer object as an unsigned 64bit value
 *
 * This has exact same functionality as @ref msgpack_get_uint() except
 * it also supports the following @ref msgpack_type_t :
 * - @ref MSGPACK_TYPE_INT64
 * - @ref MSGPACK_TYPE_UINT64
 *
 * This is useful is the actual bit-length is unknown
 *
 * @param object Integer object to return an unsigned 64bit value
 * @param buffer Buffer to hold 64bit value (must be at least 8 bytes)
 * @return Pointer to 64bit value (same pointer as supplied `buffer`)
 */
uint64_t* msgpack_get_ulong(const msgpack_object_t *object, uint64_t *buffer);

/**
 * Copy a MessagePack string object into the given buffer
 *
 * A MessagePack string object is NOT null-terminated. This function
 * copies a string object's string into the given buffer AND null-terminates the string.
 *
 * @param object MessagPacket @ref MSGPACK_TYPE_STR object
 * @param buffer Buffer to hold string data, this buffer should contain an additional byte to hold the null-terminator
 * @param max_length The maximum length of the supplied `buffer`
 * @return Pointer to populated `buffer`, this is the same pointer as the supplied `buffer`
 */
char* msgpack_get_str(const msgpack_object_t *object, char* buffer, uint16_t max_length);

/**
 * Compare a string object to a string
 *
 * This has the same functionality as `strcmp()` but accepts a @ref msgpack_object_str_t
 * as the first argument.
 *
 * @param object String msgpack object
 * @param str String to compare
 * @return <0 the first character that does not match has a lower value in `object` than in `str`
 *          0 the contents of both strings are equal
 *         >0 the first character that does not match has a greater value in `object` than in `str`
 */
int msgpack_str_cmp(const msgpack_object_t* object, const char* str);



/**
 * Pack a signed integer (8-32bits)
 *
 * This packs the supplied integer into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param value signed integer value
 * @return 0 on success
 */
int msgpack_write_int(msgpack_context_t *context, int32_t value);

/**
 * Pack an unsigned integer (8-32bits)
 *
 * This packs the supplied integer into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param value unsigned integer value
 * @return 0 on success
 */
int msgpack_write_uint(msgpack_context_t *context, uint32_t value);

/**
 * Pack a signed long integer (8-64bits)
 *
 * This packs the supplied integer into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param value signed integer value
 * @return 0 on success
 */
int msgpack_write_long(msgpack_context_t *context, int64_t value);

/**
 * Pack an unsigned long integer (8-64bits)
 *
 * This packs the supplied integer into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param value  unsigned integer value
 * @return 0 on success
 */
int msgpack_write_ulong(msgpack_context_t *context, uint64_t value);

/**
 * Pack a float
 *
 * This packs the float into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param value float value
 * @return 0 on success
 */
int msgpack_write_float(msgpack_context_t *context, float value);

/**
 * Pack a double
 *
 * This packs the supplied double into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param value double value
 * @return 0 on success
 */
int msgpack_write_double(msgpack_context_t *context, double value);

/**
 * Pack a NULL
 *
 * This packs a NULL value into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @return 0 on success
 */
int msgpack_write_nil(msgpack_context_t *context);

/**
 * Pack a boolean value
 *
 * This packs a boolean value into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param value Boolean value to write
 * @return 0 on success
 */
int msgpack_write_bool(msgpack_context_t *context, bool value);

/**
 * Pack a string value
 *
 * This packs a string value into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param str String value to write
 * @return 0 on success
 */
int msgpack_write_str(msgpack_context_t *context, const char *str);

/**
 * Begin packing a string value
 *
 * This writes the string size to the given @ref msgpack_context_t.
 * The string value MUST be immediately written after calling this function.
 * This function is useful if the string needs to be written in chunks.
 *
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param size Size of the string value that will be immediately follow
 * @return 0 on success
 */
int msgpack_write_str_marker(msgpack_context_t *context, uint32_t size);

/**
 * Pack a binary string value
 *
 * This packs a binary string value into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param data Binary string value to write
 * @param length Length of binary string
 * @return 0 on success
 */
int msgpack_write_bin(msgpack_context_t *context, const void *data, uint32_t length);

/**
 * Write a previously written @ref msgpack_context_t
 *
 * This writes the contents of a previously written @ref msgpack_context_t to the supplied `context`.
 * The previously written context MUST have been initialized with a buffer (not a writer)
 *
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param value_context Previously written context to write to this context
 * @return 0 on success
 */
int msgpack_write_context(msgpack_context_t *context, const msgpack_context_t *value_context);

/**
 * Begin packing a binary string value
 *
 * This writes the binary string size to the given @ref msgpack_context_t.
 * The binary string value MUST be immediately written after calling this function.
 * This function is useful if the binary string needs to be written in chunks.
 *
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param size Size of the binary string value that will be immediately follow
 * @return 0 on success
 */
int msgpack_write_bin_marker(msgpack_context_t *context, uint32_t size);

/**
 * Begin packing a dictionary (aka map)
 *
 * This writes the number of key/values that will go into the dictionary to the given @ref msgpack_context_t.
 * The dictionary key/values MUST be immediately written after calling this function.
 *
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @note The msgpack_write_dict_xxx APIs may be used to write the dictionary key/values.
 *
 * @note Set the `size` to `-1` to automatically count the number of elements.
 * In this case, @ref msgpack_finalize_dynamic() must be called once the elements are written.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param size Number of dictionary key/values that will immediately follow
 * @return 0 on success
 */
int msgpack_write_dict_marker(msgpack_context_t *context, int32_t size);

/**
 * Begin packing an array (aka map)
 *
 * This writes the number of values that will go into the array to the given @ref msgpack_context_t.
 * The array values MUST be immediately written after calling this function.
 *
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @note Set the `size` to `-1` to automatically count the number of elements.
 * In this case, @ref msgpack_finalize_dynamic() must be called once the elements are written.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param size Number of array values that will immediately follow
 * @return 0 on success
 */
int msgpack_write_array_marker(msgpack_context_t *context, int32_t size);

/**
 * Pack a NULL to the current dictionary
 *
 * This packs a NULL value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @return 0 on success
 */
int msgpack_write_dict_nil(msgpack_context_t *context, const char*key);

/**
 * Pack a boolean to the current dictionary
 *
 * This packs a boolean value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value Boolean value
 * @return 0 on success
 */
int msgpack_write_dict_bool(msgpack_context_t *context, const char*key, bool value);

/**
 * Pack a signed integer (8-32bits) to the current dictionary
 *
 * This packs a signed integer value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value signed integer value
 * @return 0 on success
 */
int msgpack_write_dict_int(msgpack_context_t *context, const char*key, int32_t value);

/**
 * Pack an unsigned integer (8-32bits) to the current dictionary
 *
 * This packs an unsigned integer value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value unsigned integer value
 * @return 0 on success
 */
int msgpack_write_dict_uint(msgpack_context_t *context, const char*key, uint32_t value);

/**
 * Pack a signed integer (8-64bits) to the current dictionary
 *
 * This packs a signed integer value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value signed integer value
 * @return 0 on success
 */
int msgpack_write_dict_long(msgpack_context_t *context, const char*key, int64_t value);

/**
 * Pack an unsigned integer (8-64bits) to the current dictionary
 *
 * This packs an unsigned integer value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value unsigned integer value
 * @return 0 on success
 */
int msgpack_write_dict_ulong(msgpack_context_t *context, const char*key, uint64_t value);

/**
 * Pack a float into to the current dictionary
 *
 * This packs a float value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value float value
 * @return 0 on success
 */
int msgpack_write_dict_float(msgpack_context_t *context, const char*key, float value);

/**
 * Pack a double into to the current dictionary
 *
 * This packs a double value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value double value
 * @return 0 on success
 */
int msgpack_write_dict_double(msgpack_context_t *context, const char*key, double value);

/**
 * Pack a string into to the current dictionary
 *
 * This packs a string value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value string value
 * @return 0 on success
 */
int msgpack_write_dict_str(msgpack_context_t *context, const char*key, const char *value);

/**
 * Pack a binary string into to the current dictionary
 *
 * This packs a binary string value with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value binary string value
 * @param length Length of binary string
 * @return 0 on success
 */
int msgpack_write_dict_bin(msgpack_context_t *context, const char*key, const void *value, uint32_t length);

/**
 * Pack a dictionary marker into to the current dictionary
 *
 * This packs a new dictionary marker with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @note Set the `dict_count` to `-1` to automatically count the number of elements.
 * In this case, @ref msgpack_finalize_dynamic() must be called once the elements are written.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param dict_count Number of key/value pairs that will go in new dictionary within the current dictionary
 * @return 0 on success
 */
int msgpack_write_dict_dict(msgpack_context_t *context, const char*key, int32_t dict_count);

/**
 * Pack a array marker into to the current dictionary
 *
 * This packs a new array marker with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @note Set the `array_count` to `-1` to automatically count the number of elements.
 * In this case, @ref msgpack_finalize_dynamic() must be called once the elements are written.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param array_count Number of entries that will go in new array within the current dictionary
 * @return 0 on success
 */
int msgpack_write_dict_array(msgpack_context_t *context, const char*key, int32_t array_count);

/**
 * Pack a previous written context into to the current dictionary
 *
 * This packs a previously written @ref msgpack_context_t with the corresponding dictionary key into the given @ref msgpack_context_t.
 * The @ref msgpack_context_t should have been previously initialized.
 * The previously written context MUST have been initialized with a buffer (not a writer)
 *
 * @ref msgpack_write_dict_marker() MUST have been previously called to specify the dictionary.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @param key Dictionary key of written value
 * @param value_context Previously written context to write to supplied `context`
 * @return 0 on success
 */
int msgpack_write_dict_context(msgpack_context_t *context, const char*key, const msgpack_context_t *value_context);


/**
 * Finalize a dynamic dict or array
 *
 * When calling @ref msgpack_write_dict_marker(), @ref msgpack_write_dict_array_marker(),
 * @ref msgpack_write_dict_dict(), or @ref msgpack_write_dict_array() with the "count" argument
 * set to `-1` then the number of elements written will be automatically counted as they're written.
 * Once all of the desired elements are written, this API should be called to finalize th actual count.
 *
 * @note This feature does have an overhead in that the element count always uses 16-bits.
 *       There is also an upper limit of 65535 elements that may be stored.
 *
 * @param context Previously initialize @ref msgpack_context_t
 * @return 0 on success
*/
int msgpack_finalize_dynamic(msgpack_context_t *context);


/**
 * Initialize buffered msgpack writer context
 *
 * This initializes a buffered msgpack writer context.
 * Once initialized, the @ref api_util_msgpack_pack APIs may be used
 * to populate msgpack data into the returned @ref msgpack_context_t
 * While writing data, the internal buffer will automatically be increased as necessary.
 *
 * Once all the data is populated, use @ref msgpack_buffered_writer_get_buffer() to
 * retrieve the populated buffer.
 *
 * Use @ref msgpack_buffered_writer_deinit() to cleanup the allocated @ref msgpack_context_t.
 *
 * @param[out] context_ptr Pointer to hold allocated @ref msgpack_context_t
 * @param[in] initial_length Initial length of internal buffer, this will increase as needed
 * @return 0 on success
 */
int msgpack_buffered_writer_init(msgpack_context_t **context_ptr, uint32_t initial_length);


/**
 * De-allocate buffered msgpack writer context
 *
 * This de-allocates the memory allocated by @ref msgpack_buffered_writer_init()
 *
 * This will also optionally de-allocated the internal buffer.
 * If the internal buffer is NOT de-allocated (e.g. `free_buffer` = false),
 * then @ref msgpack_buffered_writer_get_buffer() should FIRST be used to retrieve a
 * reference to the internal buffer.
 *
 * Then at a later point, @ref free() should be used to de-allocate the returned `buffer.data`.
 *
 * @param[in] context Buffered msgpack writer context to de-allocated
 * @param[in] free_buffer If true also de-allocate internal buffer, if false internal buffer is NOT de-allocated.
 * @return 0 on success
 */
int msgpack_buffered_writer_deinit(msgpack_context_t *context, bool free_buffer);


/**
 * Retrieve internal buffer of buffered msgpack writer context
 *
 * This retrieves the internal buffer used by a buffered msgpack writer.
 * The `buffer_ptr` points to the allocated buffer.
 * The `length_ptr` is the total amount of data written to the buffer.
 *
 * @param[in] context Buffered msgpack writer context
 * @param[out] buffer_ptr Pointer to hold underlying buffer
 * @param[out] length_ptr Pointer to hold size of populated underlying buffer
 */
int msgpack_buffered_writer_get_buffer(const msgpack_context_t *context, uint8_t** buffer_ptr, uint32_t* length_ptr);



#ifdef __cplusplus
}
#endif