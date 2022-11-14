#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>


#include "str_util.h"


#ifndef XOR_SWAP
// swap the values in the two given variables
// NOTE: fails when a and b refer to same memory location
#define XOR_SWAP(a,b) do\
{\
    a ^= b;\
    b ^= a;\
    a ^= b;\
} while (0)
#endif




/*************************************************************************************************/
char* str_chop(char *haystack, const char *needle)
{
    if (haystack == NULL)
    {
        return NULL;
    }
    char *end = strstr(haystack, needle);
    if (end != NULL)
    {
        *end = '\0';
        return end + strlen(needle);
    }
    return NULL;
}

/*************************************************************************************************/
int str_isempty(const char *s)
{
    return ((s == NULL) || (*s == 0));
}

/*************************************************************************************************/
int str_isspace(const char *s)
{
    while(*s != 0)
    {
        if(!isspace((uint8_t)*s++))
        {
            return 0;
        }
    }
    return 1;
}

/*************************************************************************************************/
int char_isprint(int c)
{
    return(c>=0x20 && c<=0x7E);
}

/*************************************************************************************************/
int str_isprint(const char *s)
{
    for(; *s != 0; ++s)
    {
        if(!char_isprint(*s))
        {
            return 0;
        }
    }
    return 1;
}

/*************************************************************************************************/
int char_isnum(int c)
{
    return(c >='0' && c <= '9');
}

/*************************************************************************************************/
int str_isnum(const char *s)
{
    for(; *s != 0; ++s)
    {
        if(!char_isnum(*s))
        {
            return 0;
        }
    }
    return 1;
}


/*************************************************************************************************/
char* str_reverse(char *str)
{
    if (str != NULL)
    {
        char *end = str + strlen(str) - 1;

        // walk inwards from both ends of the string,
        // swapping until we get to the middle
        while (str < end)
        {
            XOR_SWAP(*str, *end);
            str++;
            end--;
        }
    }

    return str;
}

/*************************************************************************************************/
char* str_replace_all(char *s, const char *match_chars, char replace_char)
{
    char *idx;

    for(const char *needle = match_chars; *needle != 0; ++needle)
    {
        char *ptr = s;

        if(replace_char == *needle)
        {
            continue;
        }

        loop:
        if((idx = strchr(ptr, *needle)) != NULL)
        {
            *idx = replace_char;
            ptr = idx + 1;
            goto loop;
        }
    }

    return s;
}

/*************************************************************************************************/
char* str_tolower(char *s)
{
    char *p = s;
    for (; *p; ++p)
    {
        *p = tolower((int) * p);
    }
    return s;
}

/*************************************************************************************************/
char* str_toupper(char *s)
{
    char *p = s;
    for (; *p; ++p)
    {
        *p = toupper((int) * p);
    }
    return s;
}

/*************************************************************************************************/
char* str_lstrip(char *s, const char *chars)
{
    return s + strspn(s, chars);
}

/*************************************************************************************************/
char* str_rstrip(char *s, const char *chars)
{
    char *end = s + strlen(s) - 1;
    while (end > s && strstr(chars, end))
    {
        *end-- = '\0';
    }
    return s;
}

/*************************************************************************************************/
char* str_strip(char *s, const char *chars)
{
    return str_rstrip(str_lstrip(s, chars), chars);
}

/*************************************************************************************************/
int str_parse_base(const char *s, int base, intmax_t *result, intmax_t min, intmax_t max)
{
    if (str_isempty(s))
    {
        return -1;
    }
    char *end;
    intmax_t value = strtoll(s, &end, base);
    if ((*end != 0) || value < min || value > max)
    {
        return -1;
    }

    *result = value;

    return 0;
}


/*************************************************************************************************/
int str_parse_int(const char *s, intmax_t *result, intmax_t min, intmax_t max)
{
    int base;
    if ( s[0] == '0' && (s[1] == 'x' || s[1] == 'X') )
    {
        base = 16;
    }
    else if ( s[0] == '0' && (s[1] == 'b' || s[1] == 'B') )
    {
        base = 2;
        // The function strtoll cannot deal with the binary prefix in its input, so skip past this.
        s += 2;
    }
    else
    {
        base = 10;
    }
    
    return str_parse_base(s, base, result, min, max);
}

/*************************************************************************************************/
int str_parse_hex(const char *s, intmax_t *result, intmax_t min, intmax_t max)
{
    return str_parse_base(s, 16, result, min, max);
}

/*************************************************************************************************/
int str_parse_bool(const char *onoff, bool *var)
{
    static const char* const on_vals[4] =
    {
            "1",
            "on",
            "true",
            "yes"
    };
    static const char* const off_vals[4] =
    {
            "0",
            "false",
            "no",
            "off",
    };

    for(int i = 0; i < 4; ++i)
    {
        if(strcmp(on_vals[i], onoff) == 0)
        {
            *var = true;
            return 0;
        }
    }

    for(int i = 0; i < 4; ++i)
    {
        if(strcmp(off_vals[i], onoff) == 0)
        {
            *var = false;
            return 0;
        }
    }

    return -1;
}

/*************************************************************************************************/
char* str_binary_to_hex_buffer(char *hex_str, uint32_t hex_str_max_len, const void *binary_data, uint32_t binary_data_len)
{
    const char *hex_str_end = hex_str + hex_str_max_len - 1;
    const uint8_t *binary_ptr = binary_data;
    char *hex_str_ptr = hex_str;

    for (int i = binary_data_len; i > 0; --i)
    {
        if(hex_str_ptr-2 >= hex_str_end)
        {
            break;
        }

        str_byte_to_hex(*binary_ptr, hex_str_ptr);
        hex_str_ptr += 2;
        ++binary_ptr;
    }

    *hex_str_ptr = 0;

    return hex_str;
}

/*************************************************************************************************/
int str_hex_to_binary(char *s)
{
    int i, j;
    uint8_t *binary_ptr = (uint8_t*)s;
    const int len = strlen(s);

    if((len & 0x01) != 0)
    {
        return -1;
    }

    for(i = j = 0; i < len; i += 2, j++)
    {
        const int num = str_hex_to_byte(s);
        if(num == -1)
        {
            return -1;
        }
        s += 2;
        *binary_ptr++ = (uint8_t)num;
    }

    return j;
}

/*************************************************************************************************/
char* str_binary_to_hex(void *binary_data, int binary_data_len)
{
    void *dst = binary_data;
    void *src= (uint8_t*)binary_data + binary_data_len;

    memcpy(src, dst, binary_data_len);

    str_binary_to_hex_buffer(dst, binary_data_len*2+1, src, binary_data_len);

    return binary_data;
}

/*************************************************************************************************/
static int from_hex_char(char c)
{
    if (c >= '0' && c <= '9')
    {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f')
    {
        return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F')
    {
        return 10 + (c - 'A');
    }
    return -1;
}

/*************************************************************************************************/
int str_hex_to_byte(const char *hex_str)
{
    const int hi = from_hex_char(*hex_str);
    const int lo = from_hex_char(*(hex_str+1));
    if (hi == -1 || lo == -1)
    {
        return -1;
    }
    return (hi << 4) | lo;
}

/*************************************************************************************************/
static char to_hex_char(int nibble)
{
    return (nibble < 10) ? nibble + '0' : (nibble-10) + 'A';
}


/*************************************************************************************************/
char* str_byte_to_hex(uint8_t byte, char *hex_str)
{
    const int upper = (byte >> 4);
    const int lower = (byte & 0x0F);

    hex_str[1] = to_hex_char(lower);
    hex_str[0] = to_hex_char(upper);

    return hex_str;
}
/*************************************************************************************************/
uint32_t str_hex_to_uint32(const char *hex_str)
{
    intmax_t val;

    if(str_parse_hex(hex_str, &val, 0, UINT32_MAX) == 0)
    {
        return (uint32_t)val;
    }
    return UINT32_MAX;
}

/*************************************************************************************************/
const char* int32_to_str(int32_t i, char *str)
{
    sprintf(str, "%d", (int)i);
    return str;
}

/*************************************************************************************************/
const char* uint32_to_str(uint32_t i, char *str)
{
    sprintf(str, "%u", (unsigned int)i);
    return str;
}

/*************************************************************************************************/
const char* uint32_to_padded_str(uint32_t value, char *output, uint8_t max_padding)
{
    uint8_t digits_left = max_padding;
    char *ptr = &output[max_padding];

    *ptr-- = 0;

    for(digits_left = max_padding; (value != 0) && (digits_left > 0); --digits_left)
    {
        *ptr-- = (char) (( value % 10 ) + '0');
        value = value / 10;
    }

    while(digits_left > 0 && ptr >= output)
    {
        *ptr-- = '0';
    }

    return output;
}

/*************************************************************************************************/
uint32_t str_to_uint32(const char *str)
{
    intmax_t val;

    if(str_parse_int(str, &val, 0, UINT32_MAX) == 0)
    {
        return (uint32_t)val;
    }
    return UINT32_MAX;
}

/*************************************************************************************************/
uint64_t str_to_uint64(const char *str)
{
    char *end;
    const int base = (str[0] == '0' && str[1] == 'x') ? 16 : 10;
    return strtoull(str, &end, base);
}

/*************************************************************************************************/
char* uint64_to_str(const uint64_t uint64, char *str_buffer)
{
    char *s = str_buffer;

    if(uint64 == 0)
    {
        str_buffer[0] = '0';
        str_buffer[1] = 0;
    }
    else
    {
        for(uint64_t div = uint64; div > 0; div = div / 10)
        {
            const uint32_t rem = div % 10;
            *str_buffer++  = rem + '0';
        }
        *str_buffer = 0;

        str_reverse(s);
    }

    return s;
}

/*************************************************************************************************/
const char* bool_to_str(bool v)
{
    return (v) ? "true" : "false";
}


/*************************************************************************************************
 * reverses a string 'str' of length 'len'
 */
static inline void _reverse(char *str, int len)
{
    int i=0, j=len-1, temp;
    while (i<j)
    {
        temp = str[i];
        str[i] = str[j];
        str[j] = temp;
        i++; j--;
    }
}

/*************************************************************************************************/
static inline int _pow_base10(int power)
{
    int pow = 1;

    while(power-- > 0)
    {
        pow *= 10;
    }
    return pow;
}

/*************************************************************************************************
 * Converts a given integer x to string str[].  d is the number
 * of digits required in output. If d is more than the number
 * of digits in x, then 0s are added at the beginning.
 */
static inline int _int_to_str(int x, char str[], int d)
{
    int i = 0;
    while (x)
    {
        str[i++] = (x%10) + '0';
        x = x/10;
    }

    // If number of digits required is more, then
    // add 0s at the beginning
    while (i < d)
        str[i++] = '0';

    _reverse(str, i);
    str[i] = '\0';
    return i;
}

/*************************************************************************************************/
const char* float_to_str(float f, char *str_buffer, uint8_t afterpoint)
{
    char *p = str_buffer;

    if(f < 0)
    {
        f = -f;
        p[0] = '-';
        ++p;
    }
    if(f < 1)
    {
        p[0] = '0';
        ++p;
    }
    // Extract integer part
    const int ipart = (int)f;

    // Extract floating part
    float fpart = f - (float)ipart;

    // convert integer part to string
    const int i = _int_to_str(ipart, p, 0);

    // check for display option after point
    if (afterpoint != 0)
    {
        p[i] = '.';  // add dot

        // Get the value of fraction part upto given no.
        // of points after dot. The third parameter is needed
        // to handle cases like 233.007
        fpart = fpart * _pow_base10(afterpoint);

        _int_to_str((int)fpart, p + i + 1, afterpoint);
    }

    return str_buffer;
}
