#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>

#include "Model.h"
#include "Tokenizator.h"

std::string GetPath()
{
	std::string Path;
	std::cout << "Перетащи файл: ";
	std::getline(std::cin, Path);

	if (!Path.empty() && Path.front() == '"' && Path.back() == '"')
		Path = Path.substr(1, Path.size() - 2);

	std::ifstream ReadFile(Path);
	if (!ReadFile.is_open())
	{
		std::cout << "\nошибка чтения файла\n";
		return "ОШИБКА\n";
	}

	std::stringstream ss;
	std::string Text;
	ss << ReadFile.rdbuf();
	Text = ss.str();

	ReadFile.close();

	return Text;
}

int main()
{
	std::cout << "Создаем словарь\n";
	std::string Text = GetPath();

	std::cout << "Создаем тестовые вопросы\n";
	std::string QA = GetPath();

	Tokens::Tokenizator Token;
	std::vector<std::string> DataSet = Token.CleanString(Text);
	std::vector<std::string> QASet = Token.CleanString(QA);
	
	Token.BringToMap(DataSet);
	Token.BringToMap(QASet);

	std::cout << "-----------------------------------\n";

	int MapSize = Token.WordToIndex.size();
	Models::DeepModel II(MapSize * 3, 256, MapSize);

	std::string RewriteBrain;
	std::cout << "\nПереобучить модель(+/-): ";
	std::getline(std::cin, RewriteBrain);
	if (RewriteBrain == "+")
	{
		std::cout << "\nЭтап 1: запущено обучение 3000 эпох\n";

		for (int epoch = 0; epoch < 3000; ++epoch)
		{
			for (int i = 0; i < DataSet.size() - 3; ++i)
			{
				std::vector<std::vector<float>> w;
				for (int j = 0; j < 3; ++j)
					w.push_back(Token.FloatVector(DataSet[i + j]));

				std::vector<float> Input;
				Input.reserve(MapSize * 3);
				for (int j = 0; j < 3; ++j)
					Input.insert(Input.end(), w[j].begin(), w[j].end());

				std::vector<float> Target = Token.FloatVector(DataSet[i + 3]);
				II.MachineLearning(Input, Target);
			}
		}

		II.SafeBrain("brain_base.bin");
	}
	else
		II.LoadBrain("brain_base.bin");

	RewriteBrain = "";
	std::cout << "\nПереобучить вопросы(+/-): ";
	std::getline(std::cin, RewriteBrain);
	if (RewriteBrain == "+")
	{
		std::cout << "\nЭтап 2: учимся отвечать на вопросы\n";

		II.LearnRate = 0.01f;
		for (int epoch = 0; epoch < 500; ++epoch)
		{
			for (int i = 0; i < QASet.size() - 3; ++i)
			{
				std::vector<std::vector<float>> w;
				for (int j = 0; j < 3; ++j)
					w.push_back(Token.FloatVector(QASet[i + j]));

				std::vector<float> Input;
				Input.reserve(MapSize * 3);
				for (int j = 0; j < 3; ++j)
					Input.insert(Input.end(), w[j].begin(), w[j].end());

				std::vector<float> Target = Token.FloatVector(QASet[i + 3]);
				II.MachineLearning(Input, Target);
			}
		}

		II.SafeBrain("brain_answer.bin");
	}
	else
		II.LoadBrain("brain_answer.bin");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.f, std::nextafter(1.f, 2.f));
	float Temperature = .05f;

	while (true)
	{
		std::string InputText;
		std::cout << "\n\nНачните диалог: ";
		std::getline(std::cin, InputText);

		std::vector<std::string> Context = Token.CleanString(InputText);
		if (Context.size() < 3)
		{
			std::cout << "\nнужно 3 слова\n";
			continue;
		}

		Context = { Context[Context.size() - 3], Context[Context.size() - 2], Context[Context.size() - 1] };

		std::cout << "AI вывод: ";

		for (int i = 0; i < 30; ++i)
		{
			std::vector<float> w1 = Token.FloatVector(Context[0]);
			std::vector<float> w2 = Token.FloatVector(Context[1]);
			std::vector<float> w3 = Token.FloatVector(Context[2]);

			std::vector<float> DetectWords;
			DetectWords.reserve(MapSize * 3);
			DetectWords.insert(DetectWords.end(), w1.begin(), w1.end());
			DetectWords.insert(DetectWords.end(), w2.begin(), w2.end());
			DetectWords.insert(DetectWords.end(), w3.begin(), w3.end());

			std::vector<float> Predict = II.Forward(DetectWords);

			auto it = std::max_element(Predict.begin(), Predict.end());
			float MaxValue = *it;
			int BestIndex = 0;
			
			float SumProb = 0.f;
			std::vector<float> Probabilities(Predict.size());
			for (int i = 0; i < Predict.size(); ++i)
			{
				float SubFloat = Predict[i]- MaxValue;
				SubFloat /= Temperature;
				Probabilities[i] = exp(SubFloat);
				SumProb += Probabilities[i];
			}

			for (auto& Prob : Probabilities)
				Prob /= SumProb;

			float RandomNumber = dist(gen);
			float PiggyBank = 0.f;
			for (int i = 0; i < Predict.size(); ++i)
			{
				PiggyBank += Probabilities[i];
				if (RandomNumber <= PiggyBank)
				{
					BestIndex = i;
					break;
				}
			}

			std::string NextWord = Token.FindTheWord(BestIndex);

			if (NextWord == "ВОПРОС")
				break;

			std::cout << NextWord << ' ';

			Context[0] = Context[1];
			Context[1] = Context[2];
			Context[2] = NextWord;
		}
	}

}
