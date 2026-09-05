#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>


class Tokenizator
{
public:

	std::unordered_map<std::string, int> wordToIndex;

	Tokenizator() = default;
	~Tokenizator() = default;

	std::vector<std::string> cleanString(const std::string& string)
	{
		std::string cleanText = string;
		std::vector<std::string> arrayString;

		for (auto& ch : cleanText)
			ch = std::tolower(ch);

		for (auto& ch : cleanText)
		{
			if (ch == ',' || ch == '.' ||
				ch == '!' || ch == '?' ||
				ch == '-' || ch == '=')
				ch = ' ';
		}

		std::stringstream ss(cleanText);
		std::string word;
		while (ss >> word)
			arrayString.push_back(word);

		return arrayString;
	}

	void bringToMap(const std::vector<std::string>& setWords)
	{
		for (auto& word : setWords)
		{
			if (wordToIndex.find(word) == wordToIndex.end())
			{
				int index = wordToIndex.size();
				wordToIndex[word] = index;
			}
		}
	}

	std::vector<float> floatVector(std::string inputText)
	{
		std::vector<float> result(wordToIndex.size(), 0.f);
		std::vector<std::string> words = cleanString(inputText);

		for (auto& Word : words)
		{
			auto it = wordToIndex.find(Word);
			if (it != wordToIndex.end())
				result[it->second] = 1.f;
		}

		return result;
	}

	std::string findTheWord(int ID)
	{
		for (auto& [word, idx] : wordToIndex)
		{
			if (ID == idx)
				return word;
		}
		return "";
	}
};