/*
 * read_file.h
 *
 *  Created on: May 7, 2018
 *      Author: ubuntu
 */

#ifndef READ_FILE_H_
#define READ_FILE_H_

#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>

std::string locateFile(const std::string& input, const std::vector<std::string> & directories);
std::string locateFile(const std::string& input);
void read_input(std::string filename, float* image, unsigned int n_sample, unsigned int n_channel);
//float* read_input(std::string filename, int n_sample, int n_channel);




#endif /* READ_FILE_H_ */
