#include "Include.h"
//#include "Neuron.h"

using namespace std;
void Neuron::updateInputWeights(Layer &prevLayer){

	// update connections in previous layer

	for (unsigned n  = 0; n < prevLayer.size(); ++n){

		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			//individual input magnified by gradient and train rate
			eta
			* neuron.getOutputVal()
			* m_gradient
			//also add momentum a fraction of previous delta
			+ alpha
			*oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const{

	double sum = 0.0;
	
	for (unsigned n = 0; n < nextLayer.size() -1; ++n){
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradients( const Layer &nextLayer){

	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);

}

void Neuron::calcOutputGradients (double targetVal){

	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);

}

double Neuron::transferFunction(double x){

	//tanh output -1 - 1
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){

	//tanh derivative
	return 1.0 - x*x;
}

void Neuron::feedForward(const Layer &prevLayer){

	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); ++n){

		sum += prevLayer[n].getOutputVal() *
			   prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){

	for (unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back( Connection() );
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
	eta = 0.15;
	alpha = 0.5;
}

Neuron::~Neuron(void)
{
}
