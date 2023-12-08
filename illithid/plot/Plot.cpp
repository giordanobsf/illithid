#include "Plot.h"

#include <matplot/matplot.h>

void Plot::Line(const std::vector<std::shared_ptr<Value<double> > >& y) const
{
    std::vector<double> yPlot(y.size());
    for (int i=0; i<y.size(); ++i)
    {
        yPlot[i] = y[i]->data();
    }

    matplot::plot(yPlot);
    matplot::show();
}

void Plot::Line(const std::vector<std::shared_ptr<Value<double> > >& x, const std::vector<std::shared_ptr<Value<double> > >& y) const
{
    if (x.size() != y.size())
    {
        throw std::invalid_argument("line plot with different sizes");
    }

    std::vector<double> xPlot(x.size());
    std::vector<double> yPlot(y.size());
    for (int i=0; i<x.size(); ++i)
    {
        xPlot[i] = x[i]->data();
        yPlot[i] = y[i]->data();
    }
    matplot::plot(xPlot, yPlot);
    matplot::show();
}

void Plot::Scatter(const std::vector<std::shared_ptr<Value<double> > >& x, const std::vector<std::shared_ptr<Value<double> > >& y) const
{
    if (x.size() != y.size())
    {
        throw std::invalid_argument("scatter plot with different sizes");
    }

    std::vector<double> xPlot(x.size());
    std::vector<double> yPlot(y.size());
    for (int i=0; i<x.size(); ++i)
    {
        xPlot[i] = x[i]->data();
        yPlot[i] = y[i]->data();
    }

    matplot::scatter(xPlot, yPlot);

    matplot::show();
}