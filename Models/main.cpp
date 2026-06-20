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
	Models::DeepModel II(MapSize * 3, 32, MapSize);

	if (!II.LoadBrain("brain.bin"))
	{
		std::cout << "запущено обучение 5000 эпох\n";

		for (int epoch = 0; epoch < 5000; ++epoch)
		{
			for (int i = 0; i < DataSet.size() - 3; ++i)
			{
				std::vector<float> w1 = Token.FloatVector(DataSet[i]);
				std::vector<float> w2 = Token.FloatVector(DataSet[i + 1]);
				std::vector<float> w3 = Token.FloatVector(DataSet[i + 2]);

				std::vector<float> Input;
				Input.reserve(MapSize * 3);
				Input.insert(Input.end(), w1.begin(), w1.end());
				Input.insert(Input.end(), w2.begin(), w2.end());
				Input.insert(Input.end(), w3.begin(), w3.end());

				std::vector<float> Target = Token.FloatVector(DataSet[i + 3]);
				II.MachineLearning(Input, Target);
			}
		}

		II.SafeBrain("brain.bin");
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.f, std::nextafter(1.f, 2.f));
	float Temperature = 0.05f;

	while (true)
	{
		std::cout << "\n\n";

		std::string InputText;
		std::cout << "Начните диалог: ";
		std::getline(std::cin, InputText);

		std::vector<std::string> Context = Token.CleanString(InputText);
		if (Context.size() < 3)
		{
			std::cout << "\nнужно 3 слова\n";
			continue;
		}

		std::cout << "AI вывод: " << InputText << ' ';

		for (int i = 0; i < 15; ++i)
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

			std::cout << NextWord << ' ';

			Context[0] = Context[1];
			Context[1] = Context[2];
			Context[2] = NextWord;
		}
	}

}
