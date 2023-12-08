#pragma once

#include "../engine/Value.h"

class Plot
{
public:
    void Line(const std::vector<std::shared_ptr<Value<double> > >& y) const;
    void Line(const std::vector<std::shared_ptr<Value<double> > >& x, const std::vector<std::shared_ptr<Value<double> > >& y) const;
    void Scatter(const std::vector<std::shared_ptr<Value<double> > >& x, const std::vector<std::shared_ptr<Value<double> > >& y) const;
};