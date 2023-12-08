#pragma once

#include "Module.h"

class Loss : public Module
{
public:
    std::vector<std::shared_ptr<Value<double> > > parameters() { return std::vector<std::shared_ptr<Value<double> > >(); }
    std::vector<std::shared_ptr<Value<double> > > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs) { return std::vector<std::shared_ptr<Value<double> > >(); }

    virtual std::shared_ptr<Value<double> > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs, const std::vector<std::shared_ptr<Value<double> > >& targets) const = 0;
};