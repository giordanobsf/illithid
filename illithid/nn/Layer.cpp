#include "Layer.h"

Layer::Layer(int numInputs, int numOutputs)
{
    for(int i=0; i<numOutputs; ++i)
    {
        m_neurons.push_back(std::make_shared<Neuron>(numInputs));
    }
}

std::vector<std::shared_ptr<Value<double> > > Layer::parameters()
{
    std::vector<std::shared_ptr<Value<double> > > params;
    for (auto n : m_neurons)
    {
        std::vector<std::shared_ptr<Value<double> > > neuronParams = n->parameters();
        params.insert(params.end(), neuronParams.begin(), neuronParams.end());
    }
    return params;
}

void Layer::zeroGrad()
{
    for (auto n : m_neurons)
    {
        n->zeroGrad();
    }
}

std::vector<std::shared_ptr<Value<double> > > Layer::forward(const std::vector<std::shared_ptr<Value<double> > >& inputs)
{
    std::vector<std::shared_ptr<Value<double> > > outs;
    for(int i=0; i<m_neurons.size(); ++i)
    {
        outs.push_back(m_neurons[i]->forward(inputs));
    }
    return outs;
}