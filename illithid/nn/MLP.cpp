#include "MLP.h"

MLP::MLP(int numInputs, const std::vector<int>& numOutputs)
{
    int inputs = numInputs;
    for (auto o : numOutputs)
    {
        m_layers.push_back(std::make_shared<Layer>(inputs, o));
        inputs = o;
    }
}

std::vector<std::shared_ptr<Value<double> > > MLP::forward(const std::vector<std::shared_ptr<Value<double> > >& inputs)
{
    std::vector<std::shared_ptr<Value<double> > > x = inputs;
    for(int i=0; i<m_layers.size(); ++i)
    {
        x = m_layers[i]->forward(x);
    }
    return x;
}