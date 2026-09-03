#include "MicroLLM.h"

#include "models/Tokenizator.h"

#include <QLabel>
#include <QScrollBar>
#include <QInputDialog>
#include <QFile>
#include <QMessageBox>
#include <QPushButton>
#include <QAbstractButton>
#include <QFileDialog>
#include <QTimer>

#include <thread>

MicroLLM::MicroLLM(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    ui.scrollArea->setWidgetResizable(true);
    ui.scrollArea->setFixedWidth(700);
    ui.scrollArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui.verticalLayout->setAlignment(ui.scrollArea, Qt::AlignHCenter);
    ui.scrollArea->setStyleSheet(R"(
        QScrollArea {
           border: none;
           background: transparent;
        }
        QScrollBar:vertical {
           width: 8px;
           margin: 0px 0px 0px 0px;
           background: transparent;
        }
        QScrollBar::handle:vertical {
           min-height: 30px;
           border-radius: 4px;
           background: #4A4D52;
        }
        QScrollBar::handle:vertical:hover {
           background: #686B73;
        }
        QScrollBar::handle:vertical:pressed {
           background: #888B94;
        }
        QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
           height: 0px;
           background: none;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
           background: none;
        }"
    )");

    connect(ui.scrollArea->verticalScrollBar(), &QScrollBar::rangeChanged,
        this, [this](int min, int max) {
            ui.scrollArea->verticalScrollBar()->setValue(max);
        });
    
    ui.userInput->setFixedWidth(700);
    ui.userInput->setFixedHeight(40);
    ui.userInput->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    ui.verticalLayout->setAlignment(ui.userInput, Qt::AlignHCenter);
    ui.userInput->setStyleSheet(R"(
        QLineEdit {
           font-size: 18px;
           font-family: 'Arial';
           border-radius: 9px;
           border: 1px solid #4b4b4b;
        }
        QLineEdit:disabled {
            background-color: #2b2b2b;
            color: #777777;
            border: 1px solid #333333;
        }
    )");

    ui.verticalLayoutForText->addStretch();

    QTimer::singleShot(0, this, [this](){
        initModel(getDataset());
    });

    connect(ui.userInput, &QLineEdit::returnPressed, this, &MicroLLM::sendMessage);
}

MicroLLM::~MicroLLM()
{}

void MicroLLM::createLabel(const QString& text)
{
    QLabel* messageLabel = new QLabel(text, this);
    messageLabel->setStyleSheet(R"(
        QLabel{
           font-size: 14px;
           font-family: 'Arial';
        }
    )");
    messageLabel->setWordWrap(true);

    ui.verticalLayoutForText->addWidget(messageLabel);
}

void MicroLLM::initModel(const std::vector<std::string>& dataSet)
{
    ui.userInput->setEnabled(false);

    token.bringToMap(dataSet);

    int mapSize = token.WordToIndex.size();
    AI.init(mapSize * 3, 256, mapSize);

    ui.userInput->setEnabled(true);
    ui.userInput->setFocus();

    updateMemory(dataSet);
}

void MicroLLM::updateMemory(const std::vector<std::string>& dataSet)
{
    ui.userInput->setEnabled(false);

    bool fileExists = QFile::exists("brainSafe.bin");

    QMessageBox::StandardButton choise = QMessageBox::No;

    if (fileExists)
    {
        choise = QMessageBox::question (
            this,
            "Memory initialization",
            "Saved memory found (brainSafe.bin). \n\nLoad it? (Yes - load, No - retrain)",
            QMessageBox::Yes | QMessageBox::No
        );
    }

    if (choise == QMessageBox::Yes)
    {
        AI.loadBrain("brainSafe.bin");
        createLabel("Memory successfully loaded from file");
    }
    else
    {
        createLabel("Waiting...");
        std::thread thread([this, dataSet]() {
            for (int epoch = 0; epoch < 3000; ++epoch)
            {
                for (int i = 0; i < dataSet.size() - 3; ++i)
                {
                    std::vector<std::vector<float>> w;
                    for (int j = 0; j < 3; ++j)
                        w.push_back(token.floatVector(dataSet[i + j]));

                    std::vector<float> InputUser;
                    InputUser.reserve(token.WordToIndex.size() * 3);
                    for (int j = 0; j < 3; ++j)
                        InputUser.insert(InputUser.end(), w[j].begin(), w[j].end());

                    std::vector<float> Target = token.floatVector(dataSet[i + 3]);
                    AI.machineLearning(InputUser, Target);
                }
            }
            AI.safeBrain("brainSafe.bin");

            QMetaObject::invokeMethod(this, [this]() {
                createLabel("Training complete! The model is ready to communicate");
            });
        });
        thread.detach();
    }

    ui.userInput->setEnabled(true);
    ui.userInput->setFocus();
}


void MicroLLM::sendMessage()
{
    QString text = ui.userInput->text().trimmed();

    if (text.isEmpty())
    {
        ui.userInput->setEnabled(true);
        ui.userInput->setFocus();
        return;
    }

    //createLabel(text);

    ui.userInput->clear();

    discussionWithModel(text);
}

void MicroLLM::discussionWithModel(QString text)
{
    ui.userInput->setEnabled(false);
    
    std::string line = text.toStdString();
    std::vector<std::string> context = token.cleanString(line);
    
    if (context.size() < 3)
    {
        createLabel("needs 3 words");
        ui.userInput->setEnabled(true);
        ui.userInput->setFocus();
        return;
    }

    int contextSize = context.size() - 1;
    context = { context[contextSize - 2], context[contextSize - 1], context[contextSize] };

    createLabel("\nThe chatbot predicts the word: ");

    QLabel* botMsg = new QLabel(this);
    botMsg->setStyleSheet("font-size: 14px; font-family: 'Arial'; color: #FFFFFF;");
    botMsg->setWordWrap(true);
    ui.verticalLayoutForText->addWidget(botMsg);

    QString fullText = QString::fromStdString("    " + std::string(context[contextSize - 2] + 
        ' ' + context[contextSize - 1]) + ' ' + context[contextSize] + ' ');

    for (int i = 0; i < 10; ++i)
    {
        std::vector<std::vector<float>> w;
        w.reserve(3);
        for (auto& str : context)
            w.push_back(token.floatVector(str));

        std::vector<float> detectWords;
        detectWords.reserve(token.WordToIndex.size() * 3);
        for (auto& word : w)
            detectWords.insert(detectWords.end(), word.begin(), word.end());

        std::vector<float> predict = AI.forward(detectWords);

        auto it = std::max_element(predict.begin(), predict.end());
        float maxVal = *it;
        int bestIdx = std::distance(predict.begin(), it);

        float sumProb = 0.f;
        std::vector<float> probabilities(predict.size());
        for (int i = 0; i < predict.size(); ++i)
        {
            float subFloat = predict[i] - maxVal;
            subFloat /= Temperature;
            probabilities[i] = exp(subFloat);
            sumProb += probabilities[i];
        }

        for (auto& prob : probabilities)
           prob /= sumProb;

        float randomNumber = dist(gen);
        float piggyBank = 0.f;
        for (int i = 0; i < predict.size(); ++i)
        {
            piggyBank += probabilities[i];
            if (randomNumber <= piggyBank)
            {
                bestIdx = i;
                break;
            }
        }

        std::string nextWord = token.findTheWord(bestIdx);

        fullText += QString::fromStdString(nextWord) + " ";
        botMsg->setText(fullText);

        context[0] = context[1];
        context[1] = context[2];
        context[2] = nextWord;
    }

    ui.userInput->setEnabled(true);
    ui.userInput->setFocus();
}

std::vector<std::string> MicroLLM::getDataset()
{
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("Dataset source");
    msgBox.setText("How would you like to provide the training dataset?");

    QPushButton* fileBtn    = msgBox.addButton("Choose file",   QMessageBox::ActionRole);
    QPushButton* textbtn     = msgBox.addButton("Enter text",    QMessageBox::ActionRole);
    QPushButton* cancelBtn  = msgBox.addButton("Cancel",        QMessageBox::RejectRole);

    msgBox.exec();

    if (msgBox.clickedButton() == cancelBtn || msgBox.clickedButton() == nullptr)
        return {};

    std::string rawContext = "";

    if (msgBox.clickedButton() == fileBtn)
    {
        QString filePath = QFileDialog::getOpenFileName(
            this,
            "Select database",
            "",
            "Text Files (*.txt);;All Files(*.*)"
        );

        if (!filePath.isEmpty())
        {
            QFile file(filePath);
            if (file.open(QIODevice::ReadOnly | QIODevice::Text))
            {
                QTextStream in(&file);
                rawContext = in.readAll().toStdString();
                file.close();
            }
        }
    }
    else if (msgBox.clickedButton() == textbtn)
    {
        bool ok;
        QString text = QInputDialog::getMultiLineText(
            this,
            "Enter dataset",
            "Type or paste your training text below:",
            "",
            &ok
        );

        if (ok && !text.isEmpty())
            rawContext = text.toStdString();
    }

    if (rawContext.empty())
        return {};

    return token.cleanString(rawContext);
}