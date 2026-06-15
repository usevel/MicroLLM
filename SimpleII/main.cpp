#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>

class Tokenizator
{
public:

	std::unordered_map<std::string, int> WordToIndex;

	Tokenizator() = default;
	~Tokenizator() = default;

	std::vector<std::string> CleanString(std::string& String)
	{
		std::string CleanText = String;
		std::vector<std::string> ArrayString;

		for (auto& Ch : CleanText)
		{
			if (Ch == ',' || Ch == '.' ||
				Ch == '!' || Ch == '?' || 
				Ch == '-' || Ch == '=' || 
				Ch == '+')
				Ch = ' ';
		}
			
		std::stringstream SS(CleanText);
		std::string Word;
		while (SS >> Word)
			ArrayString.push_back(Word);
		
		return ArrayString;
	}

	void BringToMap(std::vector<std::string>& SetWords)
	{
		int index = 0;

		for (auto& SetWord : SetWords)
		{
			std::vector<std::string> Words = CleanString(SetWord);

			for (auto& Word : Words)
			{
				if (WordToIndex.find(Word) == WordToIndex.end())
				{
					WordToIndex[Word] = index;
					++index;
				}
			}
		}
	}

	std::vector<float> FloatVector(std::string InputText)
	{
		std::vector<float> Result(WordToIndex.size(), 0.f);
		std::vector<std::string> Words = CleanString(InputText);

		for (auto& Word : Words)
		{
			auto it = WordToIndex.find(Word);
			if (it != WordToIndex.end())
				Result[it->second] = 1.f;
		}

		return Result;
	}
};

class SimpleII
{
public:
	std::vector<std::vector<float>> MatrixWeight;
	std::vector<float> Biases;


	SimpleII(int InSize, int OutSize) : InputSize(InSize), OutputSize(OutSize)
	{
		MatrixWeight.resize(OutputSize, std::vector<float>(InputSize, 0.f));
		Biases.resize(OutputSize, 0.f);
	}
	~SimpleII() = default;

	std::vector<float> Forward(const std::vector<float>& InputUser)
	{
		std::vector<float> Result(OutputSize, 0.f);

		for (int i = 0; i < OutputSize; ++i)
		{
			Result[i] = Biases[i];
			for (int j = 0; j < InputSize; ++j)
				Result[i] += MatrixWeight[i][j] * InputUser[j];
		}

		return Result;
	}

	void MachineLearning(const std::vector<float>& InputUser, const std::vector<float>& Target)
	{
		std::vector<float> Predict = Forward(InputUser);

		for (int i = 0; i < OutputSize; ++i)
		{
			float Error = Predict[i] - Target[i];
			Biases[i] -= (LearnRate * Error);

			for (int j = 0; j < InputSize; ++j)
			{
				MatrixWeight[i][j] -= (LearnRate * Error * InputUser[j]);
			}
		}
	}

private:
	int InputSize = 0;
	int OutputSize = 0;
	float LearnRate = 0.1f;
};



int main()
{
	std::vector<std::string> TestPhrases = {
		"привет",
		"как дела",
		"пока"
	};

	Tokenizator Token;
	Token.BringToMap(TestPhrases);

	std::cout << "Словарь:\n";
	for (auto& [word, idx] : Token.WordToIndex)
		std::cout << "[" << idx << "] " << word << '\n';

	std::cout << "\n-----------------------------------\n\n";

	SimpleII II(Token.WordToIndex.size(), 3);

	for (int i = 0; i < 1000; ++i)
	{
		II.MachineLearning(Token.FloatVector("привет"), { 1.f, 0.f, 0.f });
		II.MachineLearning(Token.FloatVector("как дела"), { 0.f, 1.f, 0.f });
		II.MachineLearning(Token.FloatVector("пока"), { 0.f, 0.f, 1.f });
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

		int BestIndex = 0;
		float MaxValue = Predict[0];
		for (int i = 0; i < Predict.size(); ++i)
		{
			if (Predict[i] > MaxValue)
			{
				MaxValue = Predict[i];
				BestIndex = i;
			}
		}

		std::cout << "AI вывод: ";
		if (BestIndex == 0)			std::cout << "И тебе привет!\n\n";
		else if (BestIndex == 1)	std::cout << "Всё как обычно!\n\n";
		else if (BestIndex == 2)	std::cout << "До скорого!\n\n";
	}

}