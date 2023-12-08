#include "nn/MLP.h"

int main(int argc, char** argv)
{
    MLP n(3, std::vector<int>{4, 4, 1}); //41 parameters

    std::vector<std::vector<std::shared_ptr<Value<double> > > > xs;
    xs.push_back(std::vector<std::shared_ptr<Value<double> > >{std::make_shared<Value<double> >(2.0), std::make_shared<Value<double> >(3.0), std::make_shared<Value<double> >(-1.0)});
    xs.push_back(std::vector<std::shared_ptr<Value<double> > >{std::make_shared<Value<double> >(3.0), std::make_shared<Value<double> >(-1.0), std::make_shared<Value<double> >(0.5)});
    xs.push_back(std::vector<std::shared_ptr<Value<double> > >{std::make_shared<Value<double> >(0.5), std::make_shared<Value<double> >(1.0), std::make_shared<Value<double> >(1.0)});
    xs.push_back(std::vector<std::shared_ptr<Value<double> > >{std::make_shared<Value<double> >(1.0), std::make_shared<Value<double> >(1.0), std::make_shared<Value<double> >(-1.0)}); 

    std::vector<std::shared_ptr<Value<double> > > ys = {std::make_shared<Value<double> >(1.0), std::make_shared<Value<double> >(-1.0), std::make_shared<Value<double> >(-1.0), std::make_shared<Value<double> >(1.0)}; 

    int numSteps = 100;
    double lr = 0.05;
    for (int k=0; k<numSteps; ++k)
    {
        std::shared_ptr<Value<double> > loss = std::make_shared<Value<double> >(0.0F); 
        // Forward pass
        for (int i=0; i<xs.size(); ++i)
        {
            std::vector<std::shared_ptr<Value<double> > > ypred = n.forward(xs[i]);
            loss = loss + ((ypred[0] - ys[i])->pow(2.0));
        }

        // Zero grad
        n.zeroGrad();

        // Backward pass
        loss->backward();    

        // Update (gradient descent)
        auto params = n.parameters();
        for (auto p : params)
        {
            p->addDouble(-lr * p->grad());
        }
        std::cout << k << " " << loss->data() << std::endl;
    }

    return 0;
}