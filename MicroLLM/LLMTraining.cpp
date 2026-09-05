#include "LLMTraining.h"
#include "MicroLLM.h"

#include <QThread>

LLMTraining::LLMTraining(
    Models::DeepModel& AI,
    Tokenizator& token,
    const std::vector<std::string>& dataSet,
    QObject* parent)
    : QObject{ parent }, m_AI{ AI }, m_token{ token }, m_dataSet{ dataSet }
{
}

void LLMTraining::startTraining()
{
    if (m_dataSet.size() < 4)
    {
        emit finished();
        return;
    }

    int totalEpoch = 3000;

    for (int epoch = 0; epoch < totalEpoch; ++epoch)
    {
        for (int i = 0; i + 3 < m_dataSet.size(); ++i)
        {
            if (QThread::currentThread()->isInterruptionRequested())
                return;

            std::vector<std::vector<float>> w;
            for (int j = 0; j < 3; ++j)
                w.push_back(m_token.floatVector(m_dataSet[i + j]));

            std::vector<float> InputUser;
            InputUser.reserve(m_token.wordToIndex.size() * 3);
            for (int j = 0; j < 3; ++j)
                InputUser.insert(InputUser.end(), w[j].begin(), w[j].end());

            std::vector<float> Target = m_token.floatVector(m_dataSet[i + 3]);
            m_AI.machineLearning(InputUser, Target);
        }

        if (epoch % 100 == 0)
            emit trainingStatus(epoch, totalEpoch);
    }
    m_AI.safeBrain("brainSafe.bin");

    emit finished();
}