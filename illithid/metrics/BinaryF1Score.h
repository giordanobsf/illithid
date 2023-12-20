#pragma once

#include "ClassificationMetric.h"

class BinaryF1Score : public ClassificationMetric
{
public:
    BinaryF1Score(const std::vector<std::shared_ptr<Value<double> > >& input, const std::vector<std::shared_ptr<Value<double> > >& target, double threshold=0.5);

    std::shared_ptr<Value<double> > value() const;

private:
    std::shared_ptr<Value<double> > m_value;
};