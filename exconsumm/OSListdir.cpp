//#include "stdafx.h"
#ifdef _WIN32
#include <windows.h>
#define OSSEP "\\"
#else
#include <glob.h>
#include <vector>
#include <string>
#include <dirent.h>
#define OSSEP "/"
#endif
#include "OSListdir.h"


OSListdir::OSListdir(void)
{
}


OSListdir::~OSListdir(void)
{
}

bool OSListdir::listdirs(std::string path, std::vector<std::string> &refvecFiles)
{
#ifdef _WIN32
    std::string strFilePath; // Filepath
    std::string strPattern; // Pattern
    std::string strExtension; // Extension
    HANDLE hFile; // Handle to file
    WIN32_FIND_DATA FileInformation; // File information


    strPattern = path + "\\*";

    hFile = ::FindFirstFile(strPattern.c_str(), &FileInformation);
    if (hFile != INVALID_HANDLE_VALUE)
    {
        do
        {
            if (FileInformation.cFileName[0] != '.')
            {
                strFilePath.erase();
                strFilePath = path + "\\" + FileInformation.cFileName;

                if (FileInformation.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                {
                    refvecFiles.push_back(strFilePath);
                }

            }
        } while (::FindNextFile(hFile, &FileInformation) == TRUE);

        // Close handle
        ::FindClose(hFile);

        DWORD dwError = ::GetLastError();
        if (dwError != ERROR_NO_MORE_FILES)
            return true;
    }
#else
    struct dirent *de;  // Pointer for directory entry

                        // opendir() returns a pointer of DIR type. 
    DIR *dr = opendir(".");

    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Could not open current directory");
        return 0;
    }

    // Refer http://pubs.opengroup.org/onlinepubs/7990989775/xsh/readdir.html
    // for readdir()
    while ((de = readdir(dr)) != NULL)
        if (de->d_type == DT_DIR)
            refvecFiles.push_back(de->d_name);

    closedir(dr);
    return 0;
#endif 
    return false;

}

bool OSListdir::allSubFiles(std::string path, std::string pattern, std::list<std::string> &reflistFiles)
{
    std::vector<std::string> subdirs;
    std::string subdir = path;
    subdir.append("\\").append(pattern);
    glob(subdir, reflistFiles);
    listdirs(path, subdirs);

    for (size_t i = 0; i < subdirs.size(); i++)
    {
        subdir = path;
        subdir += "\\";
        subdir += subdirs[i];
        subdir.append("\\").append(pattern);
        glob(subdir, reflistFiles);
        subdir = path;
        subdir += "\\";
        subdir += subdirs[i];
        allSubFiles(subdir, pattern, reflistFiles);
    }
    if (reflistFiles.size() == 0)
        glob(path, reflistFiles);
    return reflistFiles.size() > 0;
}


bool OSListdir::glob(std::string path, std::list<std::string>&refvecFiles)
{

#ifdef _WIN32
    std::string strFilePath; // Filepath
    std::string strPattern; // Pattern
    std::string strExtension; // Extension
    HANDLE hFile; // Handle to file
    WIN32_FIND_DATA FileInformation; // File information


    strPattern = path;

    hFile = ::FindFirstFile(strPattern.c_str(), &FileInformation);
    if (hFile != INVALID_HANDLE_VALUE)
    {
        do
        {
            if (FileInformation.cFileName[0] != '.')
            {
                strFilePath.erase();
                strFilePath = path.substr(0, path.find_last_of("\\") + 1).append(FileInformation.cFileName);

                if ((FileInformation.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
                {
                    refvecFiles.push_back(strFilePath);
                }
                else
                {

                }
            }
        } while (::FindNextFile(hFile, &FileInformation) == TRUE);

        // Close handle
        ::FindClose(hFile);

        DWORD dwError = ::GetLastError();
        if (dwError != ERROR_NO_MORE_FILES)
            return true;
    }

    return false;
#else
    using namespace std;
    glob_t glob_result;
    ::glob(path.c_str(), GLOB_TILDE, NULL, &glob_result);
    for (unsigned int i = 0; i<glob_result.gl_pathc; ++i) {
        refvecFiles.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return true;
#endif

}
