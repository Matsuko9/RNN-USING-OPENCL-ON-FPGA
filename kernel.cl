__kernel void rnn(__constant float *weight_ih, __global int* iter, __constant float *input, __global float* output, __constant int* dim, __constant int* hsize, __constant float* bias, __constant float* weight_hh,  __global float* hidden_state)
{

	/* w_ih x embed_out */

	for(int i=0; i<*hsize; i++)
	{
		float sum = 0.0;
		for(int j=0; j<*dim; j++)
		{

			sum += *(weight_ih + i **dim + j) * *(input + *iter**dim + j);
		}

		*(output + *iter**hsize + i) = sum;
	
	}
	
	/* w_hh x hidden */
	
	for(int i=0; i<*hsize; i++)
	{
		float sum = 0.0;
		float tmp;
		for(int j=0; j<*hsize; j++)
		{

			sum += *(weight_hh + i **hsize + j) * *(hidden_state + j);
		}

		*(output + *iter**hsize + i) += sum;
		
			
	}
	
	/* --- Tanh ---- */
	
	for(int i=0; i<*hsize; i++)
		*(output + *iter**hsize + i) = tanh(*(output + *iter**hsize + i));
	/* --- Tanh ---- */	
	
	for(int i=0; i<*hsize; i++)
		*(hidden_state + i) = *(output + *iter**hsize + i);
	for(int i=0;i<*hsize;i++)
   *(output + *iter * *hsize + i) +=*(bias + i);
	/*
	if (*iter == 0){
		for(int i=0;i<*hsize; i++)
		{

			printf("%f ", *(output + *iter**hsize + i));
		}
	}*/

}

__kernel void linear_kernel(__constant float *weights, __constant float *bias, __global float* input, __global float* output, __constant int* rows, __constant int *cols)
{
	int work_id = get_global_id(0);

	for(int i=0; i<*rows; i++)
	{
		float sum = 0.0;
		for(int j=0; j<*cols; j++)
		{
			sum += *(weights + i**cols + j) * *(input + work_id**cols + j);
		}
		*(output + work_id**rows + i) = sum;
		*(output + work_id**rows + i) += *(bias + i);
	}

	/*
	if (work_id == 1){
		for(int i=0;i<*rows; i++)
		{
			printf("%f ", *(output + work_id**rows + i));
		}
	}*/

}
