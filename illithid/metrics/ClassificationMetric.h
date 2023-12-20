#pragma once

#include "Metric.h"

class ClassificationMetric : public Metric
{
public:
    virtual std::shared_ptr<Value<double> > value() const = 0;
};