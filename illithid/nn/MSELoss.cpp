#include "MSELoss.h"

std::shared_ptr<Value<double> > MSELoss::forward(const std::vector<std::shared_ptr<Value<double> > >& inputs, const std::vector<std::shared_ptr<Value<double> > >& targets) const
{    
    if (inputs.size() != targets.size())
    {
        throw std::invalid_argument("different sizes of inputs and targets");
    }

    std::shared_ptr<Value<double> > loss = std::make_shared<Value<double> >(0.0F); 
    for (int i=0; i<targets.size(); ++i)
    {
        loss = loss + ((targets[i] - inputs[i])->pow(2.0));
    }
    
    return loss;
}