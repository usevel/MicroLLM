#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MicroLLM.h"

class MicroLLM : public QMainWindow
{
    Q_OBJECT

public:
    MicroLLM(QWidget *parent = nullptr);
    ~MicroLLM();

    void sendMessage();

private:
    Ui::MicroLLMClass ui;
};

