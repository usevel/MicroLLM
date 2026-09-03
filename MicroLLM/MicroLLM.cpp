#include "MicroLLM.h"

#include <QLabel>

MicroLLM::MicroLLM(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    ui.scrollArea->setWidgetResizable(true);
    ui.scrollArea->setFixedWidth(700);
    ui.scrollArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui.verticalLayout->setAlignment(ui.scrollArea, Qt::AlignHCenter);

    
    ui.userInput->setFixedWidth(700);
    ui.userInput->setFixedHeight(40);
    ui.userInput->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    ui.verticalLayout->setAlignment(ui.userInput, Qt::AlignHCenter);
    ui.userInput->setStyleSheet(
        "QLineEdit {"
        "   font-size: 18px;"
        "   font-family: 'Arial';"
        "   border-radius: 9px;"
        "   border: 1px solid #4b4b4b;"
        "}"
    );

    ui.verticalLayoutForText->addStretch();

    connect(ui.userInput, &QLineEdit::returnPressed, this, &MicroLLM::sendMessage);
    
}

MicroLLM::~MicroLLM()
{}

void MicroLLM::sendMessage()
{
    QString text = ui.userInput->text().trimmed();

    if (text.isEmpty())
        return;

    QLabel* messageLabel = new QLabel(text, this);
    messageLabel->setStyleSheet(
        "QLabel {"
        "   font-size: 14px;"
        "   font-family: 'Arial';"
        "}"
    );
    messageLabel->setWordWrap(true);

    ui.verticalLayoutForText->addWidget(messageLabel);

    ui.userInput->clear();
}