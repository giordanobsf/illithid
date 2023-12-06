#include "nn/MLP.h"

// int main(int argc, char** argv)
// {
    // TEST 1 +, - and *
    // std::shared_ptr<Value<double> > a = std::make_shared<Value<double> >(1.0);
    // std::shared_ptr<Value<double> > b = std::make_shared<Value<double> >(2.0);
    // std::shared_ptr<Value<double> > c = a + b;
    // std::shared_ptr<Value<double> > d = std::make_shared<Value<double> >(4.0);
    // std::shared_ptr<Value<double> > e = c * d;
    // std::shared_ptr<Value<double> > f = std::make_shared<Value<double> >(12.0);
    // std::shared_ptr<Value<double> > g = e - f;

    // std::cout << a << b << c << d << e << f << g << std::endl;
    // g->backward();
    // std::cout << a << b << c << d << e << f << g << std::endl;
    
    // TEST 2 pow
    // std::shared_ptr<Value<double> > a = std::make_shared<Value<double> >(2.0);
    // std::shared_ptr<Value<double> > b = a->pow(3.0);
    // std::cout << a << b << std::endl;
    // b->backward();
    // std::cout << a << b << std::endl;

    // TEST 3 tanh
    // std::shared_ptr<Value<double> > a = std::make_shared<Value<double> >(0.5);
    // std::shared_ptr<Value<double> > b = a->tanh();
    // std::cout << *a << *b << std::endl;
    // b->backward();
    // std::cout << *a << *b << std::endl;
// }

int main(int argc, char** argv)
{
    MLP n(3, std::vector<int>{4, 4, 1}); //41 parameters

    std::vector<std::vector<std::shared_ptr<Value<double> > > > xs;
    xs.push_back(std::vector<std::shared_ptr<Value<double> > >{std::make_shared<Value<double> >(2.0), std::make_shared<Value<double> >(3.0), std::make_shared<Value<double> >(-1.0)});
    xs.push_back(std::vector<std::shared_ptr<Value<double> > >{std::make_shared<Value<double> >(3.0), std::make_shared<Value<double> >(-1.0), std::make_shared<Value<double> >(0.5)});
    xs.push_back(std::vector<std::shared_ptr<Value<double> > >{std::make_shared<Value<double> >(0.5), std::make_shared<Value<double> >(1.0), std::make_shared<Value<double> >(1.0)});
    xs.push_back(std::vector<std::shared_ptr<Value<double> > >{std::make_shared<Value<double> >(1.0), std::make_shared<Value<double> >(1.0), std::make_shared<Value<double> >(-1.0)}); // 53 Values

    std::vector<std::shared_ptr<Value<double> > > ys = {std::make_shared<Value<double> >(1.0), std::make_shared<Value<double> >(-1.0), std::make_shared<Value<double> >(-1.0), std::make_shared<Value<double> >(1.0)}; // 57 Values

    std::shared_ptr<Value<double> > loss = std::make_shared<Value<double> >(0.0F); // 58 values
    for (int i=0; i<xs.size(); ++i)
    {
        std::vector<std::shared_ptr<Value<double> > > ypred = n.forward(xs[i]);
        loss = loss + ((ypred[0] - ys[i])->pow(2.0));
        // std::cout << ypred[0] << std::endl;
    }

    std::cout << n << std::endl;
    std::cout << "================" << std::endl;
    loss->backward();    
    std::cout << "================" << std::endl;
    std::cout << n << std::endl;

    std::cout << "LOSS: " << *loss << std::endl;
    
    return 0;
}