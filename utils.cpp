#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

float *readWeights(int rows, int col, const char* file)
{
	float *weights;

	weights = new float [rows * col];

	FILE *fp = fopen(file, "r");

	if(fp == NULL)
	{
		std::cout<<"File : "<<file<<" read failed"<<std::endl;
		exit(-1);
	}

	for(int i=0; i<rows; i++)
	{
		for(int j=0; j<col; j++)
		{
			fscanf(fp, "%f\n", weights + i*col + j);
		}
	}

	return weights;
}

void printWeights(int rows, int col, float* weights)
{
	std::cout<<"# ----- Print Weights  ----- #"<<std::endl;

	for(int i=0; i<rows; i++)
	{
		std::cout<<"Row : "<<i+1<<std::endl;
		for(int j=0; j<col; j++)
		{
			std::cout<<*(weights + i*col + j)<<",";
		}
		std::cout<<std::endl;
	}	
	std::cout<<"# ----- Print  Weights Ends ----- #"<<std::endl;	
}

float* readbias(int rows, const char* bias_file)
{
	float* bias;
	bias = new float [rows];

	FILE *fp = fopen(bias_file, "r");

	if(fp == NULL)
	{
		std::cout<<"Fully connected : bias file read failed!";
		exit(-1);
	}

	for(int i=0; i<rows; i++)
	{
		fscanf(fp, "%f\n", bias + i);
	}

	return bias;
}

void printbias(int rows, float* bias)
{

	for(int i=0; i<rows; i++)
	{
		std::cout<<"Feature : "<<i+1<<" , Bias : "<<*(bias + i)<<"\n";
	}
}
