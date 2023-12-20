#pragma once

#include <functional>
#include <iostream>
#include <set>
#include <vector>

template <class T>
class Value : public std::enable_shared_from_this<Value<T> >
{
public:
    Value(T data) : m_data(data), m_grad(0.0F)
    {
        this->m_backward = [this]() { 
        };
    }

    T data() const { return m_data; }
    double grad() const { return m_grad; }

    void backward()
    {
        std::vector<std::shared_ptr<Value<T> > > topo;
        std::set<std::shared_ptr<Value<T> > > visited;

        std::function<void(std::shared_ptr<Value<T> >)> buildTopo = [&](std::shared_ptr<Value<T> > v) {
            if(visited.find(v) == visited.end())
            {
                visited.insert(v);
                for(auto child : v->m_previous)
                {
                    buildTopo(child);
                }
                topo.push_back(v);
            }
        };

        buildTopo(std::make_shared<Value<T> >(*this));
        
        // Initialize gradient of starting node
        this->m_grad = 1;

        // Apply the chain rule in reverse topological order
        std::reverse(topo.begin(), topo.end());
        for(auto v : topo)
        {
            v->m_backward();
        }
    }

    void addDouble(double value)
    {
        m_data += value;
    }

    void zeroGrad()
    {
        m_grad = 0.0F;
    }

    std::shared_ptr<Value<T> > pow(double power)
    {
        auto out = std::make_shared<Value<T> >(std::pow(this->m_data, power));
        out->m_previous.push_back(this->shared_from_this());

        out->m_backward = [this, power, out]() { 
            this->m_grad += (power * std::pow(this->m_data, (power-1.0))) * out->m_grad;
        };
        return out;
    }

    std::shared_ptr<Value<T> > tanh()
    {
        auto out = std::make_shared<Value<T> >(std::tanh(this->m_data));
        out->m_previous.push_back(this->shared_from_this());

        out->m_backward = [this, out]() { 
            this->m_grad += (1 - std::pow(std::tanh(this->m_data), 2.0)) * out->m_grad;
        };
        return out;
    }

    std::shared_ptr<Value<T> > relu()
    {
        auto out = std::make_shared<Value<T> >(std::max(T(0.0F), this->m_data));
        out->m_previous.push_back(this->shared_from_this());

        out->m_backward = [this, out]() {
            this->m_grad += (out->m_data > 0) ? out->m_grad : 0.0F;
        };
        return out;
    }

    friend std::shared_ptr<Value<T> > operator+(const std::shared_ptr<Value<T> >& lhs, const std::shared_ptr<Value<T> >& rhs)
    {
        auto out = std::make_shared<Value<T> >(lhs->m_data + rhs->m_data);
        out->m_previous.push_back(lhs);
        out->m_previous.push_back(rhs);

        out->m_backward = [lhs, rhs, out]() {
            lhs->m_grad += out->m_grad;
            rhs->m_grad += out->m_grad;
        };

        return out;
    }

    friend std::shared_ptr<Value<T> > operator*(const std::shared_ptr<Value<T> >& lhs, const std::shared_ptr<Value<T> >& rhs)
    {
        auto out = std::make_shared<Value<T> >(lhs->m_data * rhs->m_data);
        out->m_previous.push_back(lhs);
        out->m_previous.push_back(rhs);

        out->m_backward = [lhs, rhs, out]() {
            lhs->m_grad += rhs->m_data * out->m_grad;
            rhs->m_grad += lhs->m_data * out->m_grad;
        };

        return out;
    }

    friend std::shared_ptr<Value<T> > operator-(const std::shared_ptr<Value<T> >& lhs, const std::shared_ptr<Value<T> >& rhs)
    {
        std::shared_ptr<Value<double> > negative = std::make_shared<Value<double> >(-1.0);
        auto out = lhs + (rhs*negative);
        return out;
    }

    friend bool operator==(const std::shared_ptr<Value<T> >& lhs, const std::shared_ptr<Value<T> >& rhs) 
    { 
        return lhs->m_data == rhs->m_data;
    }
    
    friend bool operator!=(const std::shared_ptr<Value<T> >& lhs, const std::shared_ptr<Value<T> >& rhs) 
    { 
        return !(lhs == rhs); 
    }

    friend std::ostream& operator<<(std::ostream& out, const Value<T>& v)
    {
        out << "Value(ptr=" << &v << ", data=" << v.m_data << ", grad=" << v.m_grad << ", children=";
        for (auto c : v.m_previous)
        {
            out << c << ",";
        }
        out << ")";
        return out;
    }

private:
    T m_data;
    double m_grad;
    std::function<void()> m_backward;

    std::vector<std::shared_ptr<Value<T> > > m_previous;
};
