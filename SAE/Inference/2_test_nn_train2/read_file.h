//
// Created by ubuntu on 18-5-8.
//

#ifndef TEST_NN_TRAIN2_READ_FILE_H
#define TEST_NN_TRAIN2_READ_FILE_H

#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>

std::string locateFile(const std::string& input, const std::vector<std::string> & directories);
std::string locateFile(const std::string& input);
void read_input(std::string filename, float* image, unsigned int n_sample, unsigned int n_channel);
//float* read_input(std::string filename, int n_sample, int n_channel);


#endif //TEST_NN_TRAIN2_READ_FILE_H
