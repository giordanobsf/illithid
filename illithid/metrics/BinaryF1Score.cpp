#include "BinaryF1Score.h"


BinaryF1Score::BinaryF1Score(const std::vector<std::shared_ptr<Value<double> > >& input, const std::vector<std::shared_ptr<Value<double> > >& target, double threshold) :
    ClassificationMetric()
{
    if (input.size() != target.size())
    {
        throw std::invalid_argument("different sizes of inputs and targets");
    }

    const char TP = 0;
    const char TN = 1;
    const char FP = 2;
    const char FN = 3;
    std::vector<int> confusionMatrix(4, 0);
    
    std::vector<std::shared_ptr<Value<double> > > binaryInput(input.size());
    std::vector<std::shared_ptr<Value<double> > > binaryTarget(target.size());
    for (int i=0; i<input.size(); ++i)
    {
        if (input[i]->data() < threshold)
        {
            binaryInput[i] = std::make_shared<Value<double> >(0.0);
        }
        else
        {
            binaryInput[i] = std::make_shared<Value<double> >(1.0);
        }
        if (target[i]->data() < threshold)
        {
            binaryTarget[i] = std::make_shared<Value<double> >(0.0);
        }
        else
        {
            binaryTarget[i] = std::make_shared<Value<double> >(1.0);
        }

        if (binaryInput[i] == std::make_shared<Value<double> >(1.0))
        {
            if (binaryInput[i] == binaryTarget[i])
            {
                confusionMatrix[TP]++;
            }
            else
            {
                confusionMatrix[FP]++;
            }
        }
        else
        {
            if (binaryInput[i] == binaryTarget[i])
            {
                confusionMatrix[TN]++;
            }
            else
            {
                confusionMatrix[FN]++;
            }
        }
    }

    double f1 = confusionMatrix[TP]/(confusionMatrix[TP]+(0.5*(confusionMatrix[FP] + confusionMatrix[FN])));
    m_value = std::make_shared<Value<double> >(f1);
}

std::shared_ptr<Value<double> > BinaryF1Score::value() const
{
    return m_value;
}