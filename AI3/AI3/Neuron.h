
#pragma once
#include "Include.h"

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron {

public:
	struct Connection{
	double weight;
	double deltaWeight;
	};
	
	~Neuron(void);
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal( double val ){ m_outputVal = val;}
	double getOutputVal(void) const { return m_outputVal;};
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients (double targetVal);
	void calcHiddenGradients( const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	
	double eta;		// [0-1] overall net training rate
	double alpha;	// [0-n] multiplier of last change
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) {return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};


