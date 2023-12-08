#pragma once

#include "Neuron.h"

class Tanh : public Neuron
{
public:
    Tanh(int numInputs);

    std::vector<std::shared_ptr<Value<double> > > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs);
};