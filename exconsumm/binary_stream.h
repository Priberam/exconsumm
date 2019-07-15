#pragma once

#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#if defined (WIN32) || defined (WIN) || defined (MSDOS)
#undef sun
#endif

namespace pba_local
{
    class binary_stream
    {
    protected:
        FILE *m_fp = NULL;

    public:
        ~binary_stream()
        {
            close();
        }
        bool open(std::string path, bool write)
        {
            const char *mode = (write) ? "wb" : "rb";
            if ((m_fp = fopen(path.c_str(), mode)) == NULL)
                return false;
            
            return true;
        }
        void close()
        {
            if (m_fp)
                fclose(m_fp);
            m_fp = NULL;
        }
        bool put(const std::string& istr)
        {
            char	out_char;
            const char *str = istr.c_str();
            do
            {
                if (*str != '\0')
                    out_char = *str ^ 0xff;
                else
                    out_char = '\0';
                if (fputc(out_char, m_fp) == EOF)
                    return 0;
            } while (*str++ != '\0');
            return 1;

        }
        bool put(uint8_t ch)
        {
            return (fputc(ch, m_fp) != EOF);
        }

        bool put(uint16_t number)
        {
            return (fputc((char)(number & 0xff), m_fp) != EOF &&
                fputc((char)((number & 0xff00) >> 8), m_fp) != EOF);
        }
        bool put(uint32_t number)
        {
            return (put((uint16_t)(number & 0xffff)) &&
                put((uint16_t)((number & 0xffff0000) >> 16)));
        }
        bool put(uint64_t number)
        {
            return (put((uint32_t)(number & 0xffffffff)) &&
                put((uint32_t)((number & 0xffffffff00000000) >> 32)));
        }
        bool put(float number)
        {
            uint32_t *n = (uint32_t *)&number;
            return put((uint32_t)*n);
        }
        bool get(std::string* str)
        {


            char	ch;
            *str = "";

            do
            {
#ifndef sun
                if ((ch = (unsigned char)getc(m_fp)) == (unsigned char)EOF)
#else
                if ((int)(ch = (unsigned char)fgetc(m_fp)) == EOF)
#endif
                    return 0;
                if (ch != '\0')
                {
                    ch ^= 0xff;
                    *str += ch;
                }
            } while (ch != '\0');
            return 1;
        }
        bool get(uint8_t *ch)
        {
#ifndef sun
            return (int)((*ch = (unsigned char)fgetc(m_fp)) != (unsigned char)EOF);
#else
            return (int)((int)(*ch = (unsigned char)fgetc(m_fp)) != EOF);
#endif
        }
        bool get(uint16_t* number)
        {
#ifndef sun
            *number = 0;
            return (fread((uint16_t*)number, sizeof(uint16_t), 1, m_fp) == 1);
#else
            unsigned short	i;

            if (fread(&i, sizeof(short), 1, m_fp) != 1)
                return 0;
            *number = (unsigned short)(((i & 0xff) << 8) + ((i & 0xff00) >> 8));
            return 1;
#endif
        }
        bool get(uint32_t* number)
        {
#ifndef sun
            return (fread((char *)number, sizeof(uint32_t), 1, m_fp) == 1);
#else
            uint32_t i;

            if (!get(&i))
                return 0;
            *number = i;
            if (!get(&i))
                return 0;
            *number += (i & 0xffff) << 16;
            return 1;
#endif
        }
        bool get(float* number)
        {
            return get((uint32_t*)number);
        }
        bool get(uint64_t *number)
        {
#ifndef sun
            return (fread((char *)number, sizeof(uint64_t), 1, m_fp) == 1);
#else
            uint32_t i;

            if (!get(&i))
                return 0;
            *number = i;
            if (!get(&i))
                return 0;
            *number += (i & 0xffffffff) << 32;
            return 1;
#endif
        }



    };
}







