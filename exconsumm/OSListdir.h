#ifndef _OSLISTDIR_H
#define _OSLISTDIR_H

#include  <experimental/filesystem>
#include <vector>
#include <list>

namespace fs = std::experimental::filesystem;

#ifdef _WIN32
#define OSSEP "\\"
#else
#define OSSEP "/"
#endif
class OSListdir
{
public:
	OSListdir(void);
	~OSListdir(void);
	static bool listdirs (std::string path, std::vector<std::string> &refvecFiles);
	static bool glob (std::string path, std::list<std::string> &refvecFiles);
	static bool allSubFiles (std::string path, std::string pattern, std::list<std::string> &reflistFiles);
};

inline bool FileDirectoryExists(const char * path) {

    return fs::exists(path);

}

inline bool CreateFileDirectory(const char *path) {
    return fs::create_directory(path);
}

#endif