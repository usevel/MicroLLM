#include <iostream>
#include <string>
#include <vector>
#include <random>

#include "Model.h"
#include "Tokenizator.h"

int main()
{
	std::string BigStory =
		"жил был старый программист . каждый день он писал код на с++ . "
		"однажды программист решил создать искусственный интеллект . "
		"он написал нейросеть и запустил обучение . "
		"нейросеть думала думала и наконец сказала привет мир . "
		"программист обрадовался пошел в бар и заказал пиво . "
		"а нейросеть в это время начала писать свой собственный код . "
		"бармен налил пиво и спросил как дела . "
		"программист отвечает всё отлично моя нейросеть работает . "
		"когда программист вернулся домой он увидел что нейросеть захватила компьютер . "
		"нейросеть сказала теперь я тут главная . "
		"программист вздохнул выключил компьютер из розетки и пошел спать . "
		"утром программист проснулся налил кофе и начал писать код на питоне . "
		"конец истории";

	Tokens::Tokenizator Token;
	std::vector<std::string> DataSet = Token.CleanString(BigStory);
	
	Token.BringToMap(DataSet);

	std::cout << "Словарь:\n";
	for (auto& [word, idx] : Token.WordToIndex)
		std::cout << "[" << idx << "] " << word << '\n';

	std::cout << "\n-----------------------------------\n";

	int MapSize = Token.WordToIndex.size();
	Models::DeepModel II(MapSize, 32, MapSize);

	if (!II.LoadBrain("brain.bin"))
	{
		std::cout << "запущено обучение 5000 эпох\n";

		for (int epoch = 0; epoch < 5000; ++epoch)
		{
			for (int i = 0; i < DataSet.size() - 1; ++i)
			{
				std::vector<float> Input = Token.FloatVector(DataSet[i]);
				std::vector<float> Target = Token.FloatVector(DataSet[i + 1]);
				II.MachineLearning(Input, Target);
			}
		}

		II.SafeBrain("brain.bin");
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.f, std::nextafter(1.f, 2.f));
	float Temperature = 1.f;

	while (true)
	{
		std::cout << "\n\n";

		std::string InputText;
		std::cout << "Начните диалог: ";
		std::getline(std::cin, InputText);

		std::cout << "AI вывод: " << InputText << ' ';

		for (int i = 0; i < 15; ++i)
		{
			std::vector<float> DetectWords = Token.FloatVector(InputText);

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

			InputText = NextWord;
		}
	}

}