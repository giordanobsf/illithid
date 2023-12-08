#pragma once

#include "Neuron.h"

class ReLU : public Neuron
{
public:
    ReLU(int numInputs);

    std::vector<std::shared_ptr<Value<double> > > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs);
};