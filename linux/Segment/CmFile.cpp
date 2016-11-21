#include "Stdafx.h"

#ifdef _WIN32
#include <shlobj.h>
#include <Commdlg.h>
#include <ShellAPI.h>
#endif

BOOL CmFile::MkDir(CStr&  _path)
{
	if(_path.size() == 0)
		return false;

	static char buffer[1024];
	strcpy(buffer, _S(_path));
	for (int i = 0; buffer[i] != 0; i ++) {
		if (buffer[i] == '\\' || buffer[i] == '/') {
			buffer[i] = '\0';
#ifdef _WIN32
			CreateDirectoryA(buffer, 0);
#else
			mkdir(buffer, ALLPERMS);
#endif
			buffer[i] = '/';
		}
	}
#ifdef _WIN32
	return CreateDirectoryA(_S(_path), 0);
#else
	return mkdir(_S(_path), ALLPERMS);
#endif
}

#ifdef _WIN32
int CmFile::GetSubFolders(CStr& folder, vecS& subFolders)
{
	subFolders.clear();
	WIN32_FIND_DATAA fileFindData;
	string nameWC = folder + "\\*";
	HANDLE hFind = ::FindFirstFileA(nameWC.c_str(), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	do {
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			subFolders.push_back(fileFindData.cFileName);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return (int)subFolders.size();
}
#else
int CmFile::GetSubFolders(CStr& folder, vecS& subFolders)
{
	subFolders.clear();

	DIR *dp = opendir(_S(folder));
	if (dp == NULL){
		cout << folder << endl;
		perror("Cannot open folder");
		return EXIT_FAILURE;
	}

	struct dirent *dirContent;
	while ((dirContent = readdir(dp)) != NULL){
		if (string(dirContent->d_name)[0] == '.')
			continue;
		struct stat st;
		lstat(dirContent->d_name,&st);
		if(S_ISDIR(st.st_mode)){
			subFolders.push_back(string(dirContent->d_name));
		}

	}

	closedir(dp);
	return (int)subFolders.size();
}
#endif

// Get image names from a wildcard. Eg: GetNames("D:\\*.jpg", imgNames);
#ifdef _WIN32
int CmFile::GetNames(CStr &nameW, vecS &names, string &dir)
{
	dir = GetFolder(nameW);
	names.clear();
	names.reserve(6000);
	WIN32_FIND_DATAA fileFindData;
	HANDLE hFind = ::FindFirstFileA(_S(nameW), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	do{
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue; // Ignore sub-folders
		names.push_back(fileFindData.cFileName);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return (int)names.size();
}

int CmFile::GetNames(CStr &nameW, vecS &names)
{
	string dir;
	return GetNames(nameW, names, dir);
}
#else
int CmFile::GetNames(CStr &nameW, vecS &names, string &dir)
{
	dir = GetFolder(nameW);
	names.clear();
	names.reserve(6000);

	DIR *dp = opendir(_S(dir));
	if (dp == NULL){
		cout << dir << endl;
		perror("Cannot open directory");
		return EXIT_FAILURE;
	}

	struct dirent *dirContent;
	while ((dirContent = readdir(dp)) != NULL){
		if (string(dirContent->d_name)[0] == '.')
			continue;
		struct stat st;
		lstat(dirContent->d_name,&st);
		if(S_ISREG(st.st_mode)){
			names.push_back(string(dirContent->d_name));
		}

	}

	closedir(dp);
	return (int)names.size();
}

int CmFile::GetNames(CStr &nameW, vecS &names)
{
	string dir = GetFolder(nameW);
	names.clear();
	names.reserve(6000);

	DIR *dp = opendir(_S(dir));
	if (dp == NULL){
		cout << dir << endl;
		perror("Cannot open directory");
		return EXIT_FAILURE;
	}

	struct dirent *dirContent;
	while ((dirContent = readdir(dp)) != NULL){		
		if (string(dirContent->d_name)[0] == '.')
			continue;
		struct stat st;
		lstat(dirContent->d_name,&st);
		
		if(S_ISREG(st.st_mode)){
			cout << string(dirContent->d_name) << " " << st.st_mode << endl;
			names.push_back(string(dirContent->d_name));
		}

	}

	closedir(dp);
	return (int)names.size();
}
#endif

int CmFile::GetNames(CStr& rootFolder, CStr &fileW, vecS &names)
{
	GetNames(rootFolder + fileW, names);
	vecS subFolders, tmpNames;
	int subNum = CmFile::GetSubFolders(rootFolder, subFolders);
	for (int i = 0; i < subNum; i++){
		subFolders[i] += "/";
		int subNum = GetNames(rootFolder + subFolders[i], fileW, tmpNames);
		for (int j = 0; j < subNum; j++)
			names.push_back(subFolders[i] + tmpNames[j]);
	}
	return (int)names.size();
}

#ifdef _WIN32
int CmFile::GetNamesNE(CStr& nameWC, vecS &names, string &dir, string &ext)
{
	int fNum = GetNames(nameWC, names, dir);
	ext = GetExtention(nameWC);
	for (int i = 0; i < fNum; i++)
		names[i] = GetNameNE(names[i]);
	return fNum;
}

int CmFile::GetNamesNE(CStr& nameWC, vecS &names)
{
	string dir, ext;
	return GetNamesNE(nameWC, names, dir, ext);
}
#else
int CmFile::GetNamesNE(CStr& nameWC, vecS &names, string &dir, string &ext)
{
	int fNum = GetNames(nameWC, names, dir);
	//ext = GetExtention(nameWC);
	for (int i = 0; i < fNum; i++)
		names[i] = GetNameNE(names[i]);
	return fNum;
}

int CmFile::GetNamesNE(CStr& nameWC, vecS &names)
{
	int fNum = GetNames(nameWC, names);
	//string ext = GetExtention(nameWC);
	for (int i = 0; i < fNum; i++)
		names[i] = GetNameNE(names[i]);
	return fNum;
}
#endif

int CmFile::GetNamesNE(CStr& rootFolder, CStr &fileW, vecS &names)
{
	int fNum = GetNames(rootFolder, fileW, names);
	int extS = GetExtention(fileW).size();
	for (int i = 0; i < fNum; i++)
		names[i].resize(names[i].size() - extS);
	return fNum;
}

vecS CmFile::loadStrList(CStr &fName)
{
	ifstream fIn(_S(fName));
	string line;
	vecS strs;
	while(getline(fIn, line) && line.size()){
#ifndef _WIN32
		int line_size = line.size();
		//avoid copying the carriage return character
		if (line[line_size-1] == '\r')
			strs.push_back(line.substr(0,line_size-1));
		else
#endif
			strs.push_back(line);
	}
	return strs;
}

bool CmFile::writeStrList(CStr &fName, const vecS &strs)
{
	FILE *f = fopen(_S(fName), "w");
	if (f == NULL)
		return false;
	for (size_t i = 0; i < strs.size(); i++)
		fprintf(f, "%s\n", _S(strs[i]));
	fclose(f);
	return true;
}
