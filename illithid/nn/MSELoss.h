#pragma once

#include "Loss.h"

class MSELoss : public Loss
{
public:
    std::shared_ptr<Value<double> > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs, const std::vector<std::shared_ptr<Value<double> > >& targets) const;
};