#pragma once

#include <QtWidgets/QMainWindow>

#include <random>

#include "ui_MicroLLM.h"

#include "models/Model.h"
#include "models/Tokenizator.h"

class MicroLLM : public QMainWindow
{
    Q_OBJECT

public:
    MicroLLM(QWidget *parent = nullptr);
    ~MicroLLM();

    void createLabel(const QString& text);
    void sendMessage();
    void initModel(const std::vector<std::string>& dataSet);
    void updateMemory(const std::vector<std::string>& dataSet);
    void discussionWithModel(QString text);
    void closeEvent(QCloseEvent* event);

    std::vector<std::string> getDataset();

private:
    Ui::MicroLLMClass ui;

    Tokenizator token;
    Models::DeepModel AI;

    std::random_device rd;
    std::mt19937 gen{ rd() };
    std::uniform_real_distribution<float> dist{ 0.f, 1.f };
    float Temperature = .8f;

    QThread* trainingThread = nullptr;
};