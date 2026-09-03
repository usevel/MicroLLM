#pragma once

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

	std::vector<std::string> cleanString(const std::string& String)
	{
		std::string CleanText = String;
		std::vector<std::string> ArrayString;

		for (auto& Ch : CleanText)
			Ch = std::tolower(Ch);

		for (auto& Ch : CleanText)
		{
			if (Ch == ',' || Ch == '.' ||
				Ch == '!' || Ch == '?' ||
				Ch == '-' || Ch == '=')
				Ch = ' ';
		}

		std::stringstream SS(CleanText);
		std::string Word;
		while (SS >> Word)
			ArrayString.push_back(Word);

		return ArrayString;
	}

	void bringToMap(const std::vector<std::string>& SetWords)
	{
		for (auto& Word : SetWords)
		{
			if (WordToIndex.find(Word) == WordToIndex.end())
			{
				int index = WordToIndex.size();
				WordToIndex[Word] = index;
			}
		}
	}

	std::vector<float> floatVector(std::string InputText)
	{
		std::vector<float> Result(WordToIndex.size(), 0.f);
		std::vector<std::string> Words = cleanString(InputText);

		for (auto& Word : Words)
		{
			auto it = WordToIndex.find(Word);
			if (it != WordToIndex.end())
				Result[it->second] = 1.f;
		}

		return Result;
	}

	std::string findTheWord(int ID)
	{
		for (auto& [Word, idx] : WordToIndex)
		{
			if (ID == idx)
				return Word;
		}
		return "";
	}
};