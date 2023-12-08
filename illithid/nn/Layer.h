#pragma once

#include "Neuron.h"

template<class T>
class Layer : public Module
{
private:
    std::vector<std::shared_ptr<T> > m_neurons;

public:
    Layer(int numInputs, int numOutputs)
    {
        for(int i=0; i<numOutputs; ++i)
        {
            m_neurons.push_back(std::make_shared<T>(numInputs));
        }
    }
    
    std::vector<std::shared_ptr<Value<double> > > parameters()
    {
        std::vector<std::shared_ptr<Value<double> > > params;
        for (auto n : m_neurons)
        {
            std::vector<std::shared_ptr<Value<double> > > neuronParams = n->parameters();
            params.insert(params.end(), neuronParams.begin(), neuronParams.end());
        }
        return params;
    }

    std::vector<std::shared_ptr<Value<double> > > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs)
    {
        std::vector<std::shared_ptr<Value<double> > > outs;
        for(int i=0; i<m_neurons.size(); ++i)
        {
            std::vector<std::shared_ptr<Value<double> > > output = m_neurons[i]->forward(inputs);
            outs.insert(outs.end(), output.begin(), output.end());
        }
        return outs;
    }

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
