#ifndef UTIL_H
#define UTIL_H
#include<string>

template <class TContainer>
inline void string_split(const std::string& str,
    TContainer * tokens,
    const std::string& delimiters = "\t") {
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);

    // Find first non-delimiter.
    std::string::size_type pos = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos) {
        // Found a token, add it to the vector.
        tokens->push_back(str.substr(lastPos, pos - lastPos));

        // Skip delimiters.
        lastPos = str.find_first_not_of(delimiters, pos);

        // Find next non-delimiter.
        pos = str.find_first_of(delimiters, lastPos);
    }
}

#endif
