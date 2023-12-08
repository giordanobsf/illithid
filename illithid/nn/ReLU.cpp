#include "ReLU.h"

ReLU::ReLU(int numInputs) : Neuron(numInputs)
{
}

std::vector<std::shared_ptr<Value<double> > > ReLU::forward(const std::vector<std::shared_ptr<Value<double> > >& inputs)
{
    // Call super class forward
    std::vector<std::shared_ptr<Value<double> > > activations = Neuron::forward(inputs);
    std::vector<std::shared_ptr<Value<double> > > out;
    for (auto activation : activations)
    {
        out.push_back(activation->relu());
    }
    return out;
}