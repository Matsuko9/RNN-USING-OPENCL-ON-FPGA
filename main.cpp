#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <string>
#include <map>
#include <iostream>
#include "Utils/utils.hpp"
#include "Utils/utils.cpp"

struct rnn
{
	int input_size;
	int hidden_size;
	const char* ih_file;
	const char* hh_file;
	const char* bias;
};

struct full_connected
{
	int rows;
	int cols;
	const char* weight_file;
	const char* bias_file;
};

int main()
{
	// Initialization //
	static const char* rnn_ih_file = "weights/simple_rnn_[0]_nl.txt";
	static const char* rnn_hh_file = "weights/simple_rnn_[1]_nl.txt";
	static const char* rnn_bias_file = "weights/simple_rnn_[2]bias_nl.txt";
	static const char* fc_weights_file = "weights/dense_weight.txt";
	static const char* fc_bias_file = "weights/dense_bias.txt";
	static const char* input_file = "weights/input_vector.txt";

	int input_size = 28;
	int hidden_size = 50;

	struct rnn rn;
	rn.input_size = input_size;
	rn.hidden_size = hidden_size;
	rn.ih_file = rnn_ih_file;
	rn.hh_file = rnn_hh_file;
	rn.bias = rnn_bias_file;

	struct full_connected fc;
	fc.rows = 10;
	fc.cols = hidden_size;
	fc.weight_file = fc_weights_file;
	fc.bias_file = fc_bias_file;

	/* OpenCL implementation */

	float *rnn_ih_weights = new float[rn.input_size * rn.hidden_size];
	float *rnn_bias = new float[rn.hidden_size];
	float *rnn_hh_weights = new float[rn.hidden_size * rn.hidden_size];
	float *liner_weights = new float[fc.rows * fc.cols];
	float *liner_bias = new float[fc.rows];
	float *input = new float[rn.input_size * rn.input_size];
	float *hidden_results = new float[rn.hidden_size * rn.input_size];
	float *linear_results = new float[rn.input_size * fc.rows];

	rnn_ih_weights = readWeights(rn.hidden_size,rn.input_size,rn.ih_file);
	rnn_hh_weights = readWeights(rn.hidden_size,rn.hidden_size,rn.hh_file);
	rnn_bias = readbias(rn.hidden_size,rn.bias);
	liner_weights = readWeights(fc.rows,fc.cols,fc.weight_file);
	liner_bias = readbias(fc.rows,fc.bias_file);
	input = readWeights(rn.input_size,rn.input_size,input_file);
    
    /* Print Weights * /
	std::cout<<"\n IH FILE : \n";
	printWeights(rn.hidden_size,rn.input_size,rnn_ih_weights);
 	std::cout<<"\n HH FILE : \n";
	printWeights(rn.hidden_size,rn.hidden_size,rnn_hh_weights);
	std::cout<<"\n Bias FILE : \n";
	printbias(rn.hidden_size,rnn_bias);
	*/

	try{
		/* --- One Time OpenCl Initializations --- */
    
	  std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		std::vector<cl::Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

		cl::Context context(devices);

		cl::CommandQueue queue = cl::CommandQueue(context,devices[0]);

    std::ifstream sourceFile("kernel.cl");
	      std::string sourceCode(
	         std::istreambuf_iterator<char>(sourceFile),
	         (std::istreambuf_iterator<char>()));
	      cl::Program::Sources source(1,
	         std::make_pair(sourceCode.c_str(),
	         sourceCode.length() + 1));

	    
	    cl::Program program = cl::Program(context, source);
	    program.build(devices);
		
		cl::Kernel rnn(program,"rnn");
		cl::Kernel linear_kernel(program,"linear_kernel");

		/* --- Put Buffers on Device  ---*/
		float *hidden_state = new float[rn.hidden_size];
		for(int i=0; i<rn.hidden_size; i++)
			*(hidden_state + i) = 0.0;

		cl::Buffer rnnihBuffer = cl::Buffer(context,CL_MEM_READ_ONLY, sizeof(float) * rn.input_size * rn.hidden_size);
		queue.enqueueWriteBuffer(rnnihBuffer,CL_TRUE,0,sizeof(float) * rn.input_size * rn.hidden_size,rnn_ih_weights);

		cl::Buffer rnnhhBuffer = cl::Buffer(context,CL_MEM_READ_ONLY,sizeof(float)*rn.hidden_size*rn.hidden_size);
		queue.enqueueWriteBuffer(rnnhhBuffer,CL_TRUE,0,sizeof(float)*rn.hidden_size*rn.hidden_size,rnn_hh_weights);

		cl::Buffer rnnBiasBuffer = cl::Buffer(context,CL_MEM_READ_ONLY,sizeof(float)*rn.hidden_size);
		queue.enqueueWriteBuffer(rnnBiasBuffer,CL_TRUE,0,sizeof(float)*rn.hidden_size,rnn_bias);

		cl::Buffer linearWeightsBuffer = cl::Buffer(context,CL_MEM_READ_ONLY,sizeof(float)*fc.rows*fc.cols);
		queue.enqueueWriteBuffer(linearWeightsBuffer,CL_TRUE,0,sizeof(float)*fc.rows*fc.cols,liner_weights);

		cl::Buffer linearBiasBuffer = cl::Buffer(context,CL_MEM_READ_ONLY,sizeof(float)*fc.rows);
		queue.enqueueWriteBuffer(linearBiasBuffer,CL_TRUE,0,sizeof(float)*fc.rows,liner_bias);

		cl::Buffer msgBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, rn.input_size*rn.input_size*sizeof(float));
		queue.enqueueWriteBuffer(msgBuffer, CL_TRUE, 0, rn.input_size*rn.input_size*sizeof(float), input);

		cl::Buffer rnnOutputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*rn.hidden_size*rn.input_size);
		queue.enqueueWriteBuffer(rnnOutputBuffer, CL_TRUE, 0, sizeof(float)*rn.hidden_size*rn.input_size,hidden_results);

		cl::Buffer hsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		queue.enqueueWriteBuffer(hsizeBuffer, CL_TRUE, 0, sizeof(int),&rn.hidden_size);

		cl::Buffer hiddenStateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float)*rn.input_size);
		queue.enqueueWriteBuffer(hiddenStateBuffer, CL_TRUE, 0, sizeof(float)*rn.input_size, hidden_state);

		cl::Buffer dimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		queue.enqueueWriteBuffer(dimBuffer, CL_TRUE, 0, sizeof(int), &rn.input_size);

		queue.finish();

		/* -- RNN LAYER --*/

		for (int i=0;i<28*28;i++)
		{
			cl::Buffer iterBuffer = cl::Buffer(context,CL_MEM_READ_ONLY,sizeof(int));
			queue.enqueueWriteBuffer(iterBuffer,CL_TRUE,0,sizeof(int),&i);

			rnn.setArg(0, rnnihBuffer);
			rnn.setArg(1,iterBuffer);
			rnn.setArg(2,msgBuffer);
			rnn.setArg(3,rnnOutputBuffer);
			rnn.setArg(4,dimBuffer);
			rnn.setArg(5,hsizeBuffer);
			rnn.setArg(6,rnnBiasBuffer);
			rnn.setArg(7,rnnhhBuffer);
			rnn.setArg(8,hiddenStateBuffer);

			cl::NDRange global(1, 1);
			cl::NDRange local(1, 1);

			queue.enqueueNDRangeKernel(rnn, cl::NullRange, global, local);
			queue.enqueueReadBuffer(rnnOutputBuffer, CL_TRUE, 0, sizeof(float)*rn.hidden_size*rn.input_size, hidden_results);

			queue.finish();
		}
 
	
   		std::cout << " RNN layer Ends \n "; 
		// --- RNN Layer Ends --//
		/* ---- Linear Layer starts ---- */

		cl::Buffer linearOutputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, rn.input_size*sizeof(float)*fc.rows);
		queue.enqueueWriteBuffer(linearOutputBuffer, CL_TRUE, 0, rn.input_size*sizeof(float)*fc.rows, linear_results);

		cl::Buffer noClassBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		queue.enqueueWriteBuffer(noClassBuffer, CL_TRUE, 0, sizeof(int), &fc.rows);

		linear_kernel.setArg(0, linearWeightsBuffer);
		linear_kernel.setArg(1, linearBiasBuffer);
		linear_kernel.setArg(2, rnnOutputBuffer);
		linear_kernel.setArg(3, linearOutputBuffer);
		linear_kernel.setArg(4, noClassBuffer);
		linear_kernel.setArg(5, hsizeBuffer);

		cl::NDRange global_linear(rn.input_size, 1);
		cl::NDRange local_linear(1, 1);

		queue.enqueueNDRangeKernel(linear_kernel, cl::NullRange, global_linear, local_linear);

		queue.enqueueReadBuffer(linearOutputBuffer, CL_TRUE, 0, rn.input_size*sizeof(float)*fc.rows, linear_results);

		queue.finish();
		/* ---- Linear Layer Ends ---- */
   std::cout << " Dense layer Ends \n "; 
	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
   
	}

  for (int i =0; i<fc.rows; i++)
    std::cout <<" "<<linear_results[i];
	

	return 0;
}
