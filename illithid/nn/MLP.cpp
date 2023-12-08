#include "MLP.h"
#include "Tanh.h"
#include "ReLU.h"

MLP::MLP(int numInputs, const std::vector<int>& numOutputs)
{
    int inputs = numInputs;
    for (auto o : numOutputs)
    {
        m_layers.push_back(std::make_shared<Layer<Tanh> >(inputs, o));
        inputs = o;
    }
}

std::vector<std::shared_ptr<Value<double> > > MLP::parameters()
{
    std::vector<std::shared_ptr<Value<double> > > params;
    for (auto l : m_layers)
    {
        std::vector<std::shared_ptr<Value<double> > > layerParams = l->parameters();
        params.insert(params.end(), layerParams.begin(), layerParams.end());
    }
    return params;
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