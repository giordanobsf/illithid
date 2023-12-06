#pragma once

#include <vector>
#include "../engine/Value.h"

class Neuron
{
private:
    std::vector<std::shared_ptr<Value<double> > > m_weights;
    std::shared_ptr<Value<double> > m_bias;

public:
    Neuron(int numInputs);
    
    std::shared_ptr<Value<double> > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs);

    friend std::ostream& operator<<(std::ostream& out, const Neuron& v)
    {
        out << "Neuron(bias=" << *(v.m_bias) << ", weights=(";
        for(auto w : v.m_weights)
        {
            out << *w << ", ";
        }
        out << "))";
        return out;
    }
};
