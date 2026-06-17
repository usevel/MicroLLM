#include <iostream>
#include <string>
#include <vector>

#include "Model.h"
#include "Tokenizator.h"

int main()
{
	std::vector<std::string> TestPhrases = {
		"зашел", "программист", "в", "бар", "и", "говорит",
		"налейте", "мне", "пиво",
		"а", "бармен", "ему", "отвечает",
		"у", "нас", "нет", "пива", "есть", "только", "кофе",
		"программист", "вздохнул", "и", "пошел", "писать", "код", "на", "с++"
	};

	Tokens::Tokenizator Token;
	Token.BringToMap(TestPhrases);

	std::cout << "Словарь:\n";
	for (auto& [word, idx] : Token.WordToIndex)
		std::cout << "[" << idx << "] " << word << '\n';

	std::cout << "\n-----------------------------------\n";

	int MapSize = Token.WordToIndex.size();
	Models::DeepModel II(MapSize, 16, MapSize);

	for (int epoch = 0; epoch < 5000; ++epoch)
	{
		for (int i = 0; i < TestPhrases.size() - 1; ++i)
		{
			std::vector<float> Input = Token.FloatVector(TestPhrases[i]);
			std::vector<float> Target = Token.FloatVector(TestPhrases[i + 1]);
			II.MachineLearning(Input, Target);
		}
	}

	while (true)
	{
		std::cout << "\n\n";

		std::string InputText;
		std::cout << "Начните диалог: ";
		std::getline(std::cin, InputText);

		std::cout << "AI вывод: " << InputText << ' ';

		for (int i = 0; i < 6; ++i)
		{
			std::vector<float> DetectWords = Token.FloatVector(InputText);

			std::vector<float> Predict = II.Forward(DetectWords);

			auto it = std::max_element(Predict.begin(), Predict.end());
			int BestIndex = std::distance(Predict.begin(), it);
			std::string NextWord = Token.FindTheWord(BestIndex);

			std::cout << NextWord << ' ';

			InputText = NextWord;
		}
	}

}