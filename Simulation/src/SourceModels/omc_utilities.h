#ifndef OMC_UTILITIES_H
#define OMC_UTILITIES_H

#include <string>
#include <sstream>
#include <vector>

/******************************************************************************/
/* Timing utilities. If OpenMP is enabled it calculates the wall time through 
 omp_get_wtime() function. Otherwise, it calculates CPU time through the clock() 
 function, available in time.h library. */

float omc_get_time();
/******************************************************************************/

/******************************************************************************/
/* A simple C/C++ class to parse input files and return requested key value 
https://github.com/bmaynard/iniReader */

#define BUFFER_SIZE 256
#define INPUT_PAIRS 80
#define INPUT_EXT ".inp"  // extension of input files

/* Parse a configuration file */
extern void parseInputFile(char *file_name);

/* Copy the value of the selected input item to the char pointer */
extern int getInputValue(char *dest, const char *key);

/* Returns nonzero if line is a string containing only whitespace or is empty */
extern int lineBlack(char *line);

/* Remove white spaces from string str_untrimmed and saves the results in
 str_trimmed. Useful for string input values, such as file names */
extern void removeSpaces(char* str_trimmed, const char* str_untrimmed);

struct inputItems {
    char key[BUFFER_SIZE];
    char value[BUFFER_SIZE];
};

extern struct inputItems input_items[];     // key,value pairs
extern int input_idx;                       // number of key,value pair

/******************************************************************************/

template <typename T>
inline T convertStringToNumber(const std::string& str) {
    std::istringstream iss(str);
    T number;
    iss >> number;
    if (iss.fail()) {
        throw std::invalid_argument("Invalid conversion from string to number");
    }
    return number;
}

inline std::vector<std::string> splitString(const std::string& str) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;

    while (std::getline(ss, item, ',')) {
        result.push_back(item);
    }

    return result;
}

template <typename T>
inline std::vector<T> getValues(const std::string& str, int n) {
    auto substrs = splitString(str);
    if (substrs.size() != n) {
        throw std::invalid_argument("Invalid number of values");
    }
    std::vector<T> values;
    for (int i = 0; i < n; ++i) {
        values.push_back(convertStringToNumber<T>(substrs[i]));
    }
    return values;
}


extern bool verbose_flag;

#endif