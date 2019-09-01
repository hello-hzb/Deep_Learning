/*
 * read_file.cpp
 *
 *  Created on: May 7, 2018
 *      Author: ubuntu
 */


#include "read_file.h"

std::string locateFile(const std::string& input, const std::vector<std::string> & directories)
{
    std::string file;
    const int MAX_DEPTH{10};
    bool found{false};
    for (auto &dir : directories)
    {
        file = dir + input;
        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(file);
            found = checkFile.is_open();
            if (found) break;
            file = "../" + file;
        }
        if (found) break;
        file.clear();
    }

    assert(!file.empty() && "Could not find a file due to it not existing in the data directory.");
    return file;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"/home/nvidia/my_file/sae_cpp/hyper_data/"};
    return locateFile(input,dirs);
}


void read_input(std::string filename, float* image, unsigned int n_sample, unsigned int n_channel)
{
    unsigned int total_number = n_sample*n_channel;
    std::cout << "Read data from "<<filename<<"..."<< std::endl;
    std::ifstream imagefile(filename);

    if(!imagefile.is_open())
    {
        std::cout<<" Open file faild"<<std::endl;
    }
    else
    {
        for (int i = 0; i < total_number; i++)
        {
            imagefile >> image[i];
        }
        std::cout<<"It's done."<<std::endl;
    }
    imagefile.close();
}



