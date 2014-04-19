
#pragma once

#include "Include.h"
//#include "Neuron.h"

class Net
{
public:

	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResults(std::vector<double> &resultVals) const;
	~Net(void);
private:
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
	std::vector<Layer> m_layers; //_layers[laynum][neuronnum]
};

