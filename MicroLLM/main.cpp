#include "MicroLLM.h"
#include <QtWidgets/QApplication>
#include <QFile>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QString qssPath = QCoreApplication::applicationDirPath() + "/style.qss";
    QFile style(qssPath);
    
    if (!style.exists())
        style.setFileName("style.qss");

    if (style.open(QFile::ReadOnly | QFile::Text))
    {
        QString styleSheet = style.readAll();
        app.setStyleSheet(styleSheet);
        style.close();
    }
    else
        qWarning() << "Warning: style.qss file not found";

    MicroLLM window;
    window.show();
    return app.exec();
}