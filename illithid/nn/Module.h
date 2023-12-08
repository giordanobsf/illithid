#pragma once

#include "../engine/Value.h"
#include <vector>

class Module {
public:
    void zeroGrad()
    {
        auto params = this->parameters();
        for (auto p : params)
        {
            p->zeroGrad();
        }
    }

    virtual std::vector<std::shared_ptr<Value<double> > > parameters() = 0;
    virtual std::vector<std::shared_ptr<Value<double> > > forward(const std::vector<std::shared_ptr<Value<double> > >& inputs) = 0;
};