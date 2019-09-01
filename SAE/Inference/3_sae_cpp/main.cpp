#include <iostream>
#include <vector>
#include <chrono>
//#include <stdio.h>
//#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include "read_file.h"
#include "variable.h"
//#define N_SAMPLES 8
#define N_CHANNELS 126
#define stack_size 3
#define my_filename "/home/nvidia/my_file/sae_cpp/hyper_data/plane.txt"

unsigned int hidden_size[stack_size]={166, 20, 166};

/*leaky_relu激活函数
 *如果要使用relu函数，调用时候设置alpha为0即可
 */
float my_leaky_relu(float x, float alpha)
{
    return (x>0)?(x):(alpha*x);
}

/**全连接网络
 * batch_size待测数据的个数
 * input_size输入数据的通道数
 * hidden_size隐藏层节点数
 */
void dense_layer(float inputs[],
                   float weight[],
                   float bias[],
                   float result[],
                   float alpha,
                   unsigned int batch_size,
                   unsigned int input_size,
                   unsigned int hidden_size)
{
    for(unsigned int i=0;i<batch_size;i++)
    {
        for(unsigned int j=0;j<hidden_size;j++)
        {
            float sum=0.0;
            for(unsigned int k=0;k<input_size;k++)
            {
                sum=sum+inputs[i*input_size+k]*weight[k*hidden_size+j];
            }
            result[i*hidden_size+j] = my_leaky_relu(sum+bias[j], alpha);//完成行列相乘求和后可以直接加上权重并激活，少一次循环
        }
    }
}

/**数据对比函数，转成hls，可以屏蔽
 * 比较a和b的差别，使用MSE来衡量
 * num_element是所有数据的个数
 */
void difference_compare(float a[], float b[], unsigned int num_element)
{

    float sum=0.0;
    for(unsigned int i=0;i<num_element;i++)
    {
        float tmp = a[i]-b[i];
        sum+=tmp*tmp;
    }
    std::cout<<"MSE is: "<<sum<<std::endl;
}

int main(void) {
    std::cout << "Hello, World!" << std::endl;
    float* image=(float*)malloc(sizeof(float)*100000*N_CHANNELS);
    //read_input(my_filename, image, 10000, N_CHANNELS);
	unsigned int N_SAMPLES =4;
	for(int i=0;i<15;i++)
	{
		N_SAMPLES = N_SAMPLES * 2;
		for(unsigned int j=0;j<N_SAMPLES*126;j++)
		{
			image[j]=0.1;
		}
		std::cout<<"n_sample is :"<<N_SAMPLES<<std::endl;
		float total = 0.0, ms;
		std::cout<<"Start Counting"<<std::endl;
		auto t_start = std::chrono::high_resolution_clock::now();	//
		//第一层隐藏层计算
		float* hidden0_out=(float*)malloc(sizeof(float)*N_SAMPLES*hidden_size[0]);
		dense_layer(image, hidden0_weight, hidden0_bias, hidden0_out, 0.2, N_SAMPLES, N_CHANNELS, hidden_size[0]);

		//第二层隐藏层输出
		float* hidden1_out=(float*)malloc(sizeof(float)*N_SAMPLES*hidden_size[1]);
		dense_layer(hidden0_out, hidden1_weight, hidden1_bias, hidden1_out, 0.2, N_SAMPLES, hidden_size[0], hidden_size[1]);

		//第三层隐藏层输出
		float* hidden2_out=(float*)malloc(sizeof(float)*N_SAMPLES*hidden_size[2]);
		dense_layer(hidden1_out, hidden2_weight, hidden2_bias, hidden2_out, 0.2, N_SAMPLES, hidden_size[1], hidden_size[2]);

		//输出层结果输出
		float* reconstruct=(float*)malloc(sizeof(float)*N_SAMPLES*N_CHANNELS);
		dense_layer(hidden2_out, output_weight, output_bias, reconstruct, 0.2, N_SAMPLES, hidden_size[2], N_CHANNELS);
		auto t_end = std::chrono::high_resolution_clock::now();		//获得结束时间
		ms = std::chrono::duration<float, std::milli>(t_end - t_start).count(); //计算耗时
		total += ms;
		std::cout<<"Stop Counting, "<<"Time is: "<<total<<"ms"<<std::endl;
		//输出层结果对比

		free(hidden0_out);
		free(hidden1_out);
		free(hidden2_out);
		free(reconstruct);
	}
	free(image);

    return 0;
}
