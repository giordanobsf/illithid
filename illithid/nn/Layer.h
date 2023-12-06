#pragma once

#include "Neuron.h"

class Layer
{
private:
    std::vector<std::shared_ptr<Neuron> > m_neurons;

public:
    Layer(int numInputs, int numOutputs);
    
    std::vector<std::shared_ptr<Value<double> > > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs);

    friend std::ostream& operator<<(std::ostream& out, const Layer& v)
    {
        out << "Layer(neurons=(";
        for(auto n : v.m_neurons)
        {
            out << *n << ", ";
        }
        out << "))";
        return out;
    }
};
