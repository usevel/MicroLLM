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

	Tokens::Tokenizator Token;
	std::vector<std::string> DataSet = Token.CleanString(Text);
	
	Token.BringToMap(DataSet);

	std::cout << "-----------------------------------\n";

	int MapSize = Token.WordToIndex.size();
	//int EmbeddingSize = 16;
	int ContextWords = 3;
	Models::DeepModel II(MapSize * ContextWords, 256, MapSize);

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

				std::vector<float> InputUser;
				InputUser.reserve(MapSize * 3);
				for (int j = 0; j < 3; ++j)
					InputUser.insert(InputUser.end(), w[j].begin(), w[j].end());

				std::vector<float> Target = Token.FloatVector(DataSet[i + 3]);
				II.MachineLearning(InputUser, Target);
			}
		}

		II.SafeBrain("brain_base.bin");
	}
	else
		II.LoadBrain("brain_base.bin");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.f, std::nextafter(1.f, 2.f));
	float Temperature = .8f;

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

		for (int i = 0; i < 10; ++i)
		{
			std::vector<std::vector<float>> w;
			for (int j = 0; j < 3; ++j)
				w.push_back(Token.FloatVector(Context[j]));

			std::vector<float> DetectWords;
			DetectWords.reserve(MapSize * 3);
			for (int j = 0; j < 3; ++j)
				DetectWords.insert(DetectWords.end(), w[j].begin(), w[j].end());

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

			std::cout << NextWord << ' ';

			Context[0] = Context[1];
			Context[1] = Context[2];
			Context[2] = NextWord;
		}
	}

}
