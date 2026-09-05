#pragma once

#include <QtWidgets/QMainWindow>
#include <QObject>

#include "models/Model.h"
#include "models/Tokenizator.h"

class LLMTraining : public QObject
{
    Q_OBJECT
public:
    LLMTraining(
        Models::DeepModel& AI, 
        Tokenizator& token, 
        const std::vector<std::string>& dataSet, 
        QObject* parent = nullptr);

    ~LLMTraining() = default;

public slots:
    void startTraining();

signals:
    void trainingStatus(int epoch, int totalEpoch);
    void finished();

private:
    Models::DeepModel& m_AI;
    Tokenizator& m_token;
    const std::vector<std::string> m_dataSet;
};