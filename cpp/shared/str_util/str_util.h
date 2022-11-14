#pragma once


#include <stdint.h>
#include <stdbool.h>


#ifdef __cplusplus
extern "C" {
#endif


/**
 * Helper to find an occurrence of a delimiter string,
 * insert '\0' in its place and return string after
 * the delimiter e.g.
 *     if char s[] = "foo://bar";
 *     - strchop(s, "://") returns "bar"
 *     - s becomes "foo"
 */
char* str_chop(char *haystack, const char *needle);

/**
 * Check if string is non-null and non-empty.
 */
int str_isempty(const char *s);

/**
 * Check is string is all spaces
 */
int str_isspace(const char *s);

/**
 * Check is string is all spaces
 */
int str_isprint(const char *s);

/**
 * Check is string contains all numeric characters
 */
int str_isnum(const char *s);

/**
 * Reverse characters in string
 */
char* str_reverse(char *s);

/**
 * Replace all matching characters in string with replacement character
 */
char* str_replace_all(char *s, const char *match_chars, char replace_char);

/**
 * Convert null-terminated string to lower case.
 * ASCII charset only.
 */
char* str_tolower(char *s);

/**
 * Convert null-terminated string to upper case.
 * ASCII charset only.
 */
char* str_toupper(char *s);

/**
 * Strip string from the left.
 * Returns pointer into the input string.
 */
char* str_lstrip(char *s, const char *chars);

/**
 * Strip string from the right.
 * Modified in place.
 */
char* str_rstrip(char *s, const char *chars);

/**
 * Combination of strip left + right.
 */
char* str_strip(char *s, const char *chars);

/**
 * Parse based integer and check if it's in bounds [min, max].
 */
int str_parse_base(const char *s, int base, intmax_t *result, intmax_t min, intmax_t max);

/**
 * Parse integer and check if it's in bounds [min, max].
 * If string value starts with '0x' then parse value as hex string.
 */
int str_parse_int(const char *s, intmax_t *result, intmax_t min, intmax_t max);

/**
 * Parse hexadecimal integer and check if it's in bounds [min, max].
 */
int str_parse_hex(const char *s, intmax_t *result, intmax_t min, intmax_t max);

/**
 * Parse string to boolean value
 */
int str_parse_bool(const char *onoff, bool *var);

/**
 * Convert binary data to hex string and put into destination buffer
 */
char* str_binary_to_hex_buffer(char *hex_str, uint32_t hex_str_max_len, const void *binary_data, uint32_t binary_data_len);


/**
 * Destructively convert hex string to binary
 * Returns number of bytes parsed or -1 on error.
 * NOTE: The supplied input string is converted to binary **in-place**
 */
int str_hex_to_binary(char *s);

/**
 * Destructively convert binary data into hex string
 * The input binary_data buffer length MUST be at least binary_data_len*2 + 1
 * as the covnersion is destructive and done in-place
 */
char* str_binary_to_hex(void *binary_data, int binary_data_len);

/**
 * Convert hex string character to byte
 */
int str_hex_to_byte(const char *hex_str);

/**
 * Convert a byte to a padded hex char (XX)
 */
char* str_byte_to_hex(uint8_t byte, char *hex_str);

/**
 * Convert hex string to uint32
 */
uint32_t str_hex_to_uint32(const char *hex_str);

/**
 * Convert int32 to string
 */
const char* int32_to_str(int32_t i, char *str);

/**
 * Convert uint32 to string
 */
const char* uint32_to_str(uint32_t i, char *str);

/**
 * Convert uint32 to string with '0' padding
 */
const char* uint32_to_padded_str(uint32_t value, char *output, uint8_t max_padding);

/**
 * Convert integer string to uint32
 * If string starts with '0x' parse has hex
 */
uint32_t str_to_uint32(const char *str);

/**
 * Convert integer string uint64
 * If string starts with '0x' parse has hex
 */
uint64_t str_to_uint64(const char *str);

/**
 * Convert uint64 to string integer
 */
char* uint64_to_str(const uint64_t uint64, char *str_buffer);


/**
 * Convert float to string (true/false)
 */
const char* bool_to_str(bool v);

/**
 * Convert a float-point number to its string representation
 *
 * @param f Floating-point value
 * @param str_buffer Buffer to hold string representation
 * @param afterpoint Number of digits to print AFTER the decimal point
 * @return String value (same pointer as supplied `str_buffer` )
 */
const char* float_to_str(float f, char *str_buffer, uint8_t afterpoint);


#ifdef __cplusplus
}
#endif