#pragma once

#include "Layer.h"

class MLP
{
private:
    std::vector<std::shared_ptr<Layer> > m_layers;

public:
    MLP(int numInputs, const std::vector<int>& numOutputs);

    std::vector<std::shared_ptr<Value<double> > > parameters();
    void zeroGrad();

    std::vector<std::shared_ptr<Value<double> > > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs);

    friend std::ostream& operator<<(std::ostream& out, const MLP& v)
    {
        out << "MLP(layers=(";
        for(auto l : v.m_layers)
        {
            out << *l << ", ";
        }
        out << "))";
        return out;
    }
};
