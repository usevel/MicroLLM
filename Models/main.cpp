#include <iostream>
#include <string>
#include <vector>

#include "Model.h"
#include "Tokenizator.h"

int main()
{
	std::vector<std::string> TestPhrases = {
		"фильм плохой",
		"фильм хороший",
		"не плохой",
		"не хороший"
	};

	Tokens::Tokenizator Token;
	Token.BringToMap(TestPhrases);

	std::cout << "Словарь:\n";
	for (auto& [word, idx] : Token.WordToIndex)
		std::cout << "[" << idx << "] " << word << '\n';

	std::cout << "\n-----------------------------------\n\n";

	Models::DeepModel II(Token.WordToIndex.size(), 8, 2);

	for (int i = 0; i < 5000; ++i)
	{
		II.MachineLearning(Token.FloatVector("фильм плохой"),	{ 0.f, 1.f });
		II.MachineLearning(Token.FloatVector("фильм хороший"),	{ 1.f, 0.f });
		II.MachineLearning(Token.FloatVector("не плохой"),		{ 1.f, 0.f });
		II.MachineLearning(Token.FloatVector("не хороший"),		{ 0.f, 1.f });
	}

	while (true)
	{
		std::string InputText;
		std::cout << "Начните диалог: ";
		std::getline(std::cin, InputText);

		std::vector<float> DetectWords = Token.FloatVector(InputText);
		std::cout << "Нейродетект: [ ";
		for (auto& Fl : DetectWords)
			std::cout << Fl << ' ';
		std::cout << "]\n";

		std::vector<float> Predict = II.Forward(DetectWords);

		std::cout << "AI вывод: ";
		if (Predict[0] > Predict[1])		std::cout << "Хороший\n\n";
		else if (Predict[0] < Predict[1])	std::cout << "Плохой\n\n";
	}

}