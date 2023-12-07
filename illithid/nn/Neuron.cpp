#include <random>
#include <cmath>    
#include "Neuron.h"

Neuron::Neuron(int numInputs)
{
    // Create a random engine
    std::random_device rd;
    std::mt19937 eng(rd());

    // Create a distribution in the range [-1, 1]
    std::uniform_real_distribution<> distr(-1, 1);

    for(int i=0; i<numInputs; ++i)
    {
        m_weights.push_back(std::make_shared<Value<double> >(distr(eng)));
    }

    m_bias = std::make_shared<Value<double> >(distr(eng));
}

std::vector<std::shared_ptr<Value<double> > > Neuron::parameters()
{
    std::vector<std::shared_ptr<Value<double> > > params;
    params.push_back(m_bias);
    params.insert(params.end(), m_weights.begin(), m_weights.end());
    return params;
}

void Neuron::zeroGrad()
{
    m_bias->zeroGrad();
    for (auto w : m_weights)
    {
        w->zeroGrad();
    }
}

std::shared_ptr<Value<double> > Neuron::forward(const std::vector<std::shared_ptr<Value<double> > >& inputs)
{
    if(inputs.size() != m_weights.size())
    {
        throw std::invalid_argument("wrong inputs");
    }

    std::shared_ptr<Value<double> >activation = m_weights[0] * inputs[0];
    for(int i=1; i<m_weights.size(); ++i)
    {
        activation = activation + (m_weights[i] * inputs[i]);
    }
    activation = activation + m_bias;
    std::shared_ptr<Value<double> > out = activation->tanh();
    // std::shared_ptr<Value<double> > out = activation->relu();
    return out;
}