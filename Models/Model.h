#pragma once

#include <vector>
#include <numeric>
#include <cstdlib>
#include <fstream>

namespace Models
{
	class SimpleModel
	{
	public:
		std::vector<std::vector<float>> MatrixWeight;
		std::vector<float> Biases;


		SimpleModel(int InSize, int OutSize) : InputSize(InSize), OutputSize(OutSize)
		{
			MatrixWeight.resize(OutputSize, std::vector<float>(InputSize, 0.f));
			Biases.resize(OutputSize, 0.f);
		}
		~SimpleModel() = default;

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
					MatrixWeight[i][j] -= (LearnRate * Error * InputUser[j]);
			}
		}

	private:
		int InputSize = 0;
		int OutputSize = 0;
		float LearnRate = 0.1f;
	};



	class DeepModel
	{
	public:
		std::vector<std::vector<float>> MatrixWeight1;
		std::vector<float> Biases1;

		std::vector<std::vector<float>> MatrixWeight2;
		std::vector<float> Biases2;

		std::vector<float> HiddenRaw;
		std::vector<float> Hidden;
		std::vector<float> Predict;

		//std::vector<std::vector<float>> EmbeddingTable;

		float LearnRate = 0.05f;

		DeepModel(int InSize, int HidSize, int OutSize)
			: InputSize{ InSize }, HiddenSize{ HidSize }, OutputSize { OutSize }
		{
			Biases1.resize(HiddenSize, 0.f);
			Biases2.resize(OutputSize, 0.f);

			MatrixWeight1.resize(HiddenSize, std::vector<float>(InputSize));
			MatrixWeight2.resize(OutputSize, std::vector<float>(HiddenSize));

			for (int i = 0; i < HiddenSize; ++i)
				for (int j = 0; j < InputSize; ++j)
					MatrixWeight1[i][j] = (float(rand() % 100) / 1000.f) - 0.05f;

			for (int i = 0; i < OutputSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					MatrixWeight2[i][j] = (float(rand() % 100) / 1000.f) - 0.05f;

			HiddenRaw.resize(HiddenSize, 0.f);
			Hidden.resize(HiddenSize, 0.f);
			Predict.resize(OutputSize, 0.f);

			/*
			EmbeddingTable.resize(OutputSize, std::vector<float>(EmbeddingSize));

			for (int i = 0; i < OutputSize; ++i)
				for (int j = 0; j < EmbeddingSize; ++j)
					EmbeddingTable[i][j] = (float(rand() % 100) / 1000.f) - 0.05f;
			*/
		}
		~DeepModel() = default;

		std::vector<float> Forward(const std::vector<float>& InputUser)
		{
			for (int i = 0; i < HiddenSize; ++i)
			{
				HiddenRaw[i] = Biases1[i];
				for (int j = 0; j < InputSize; ++j)
					HiddenRaw[i] += MatrixWeight1[i][j] * InputUser[j];

				Hidden[i] = (HiddenRaw[i] < 0 ? 0.f : HiddenRaw[i]);
			}

			for (int i = 0; i < OutputSize; ++i)
			{
				Predict[i] = Biases2[i];
				for (int j = 0; j < HiddenSize; ++j)
					Predict[i] += MatrixWeight2[i][j] * Hidden[j];
			}

			return Predict;
		}

		void MachineLearning(const std::vector<float>& InputUser, const std::vector<float>& Target)
		{
			/*
			std::vector<float> InputUser;
			InputUser.reserve(EmbeddingSize * 3);
			for (int i = 0; i < InputIDs.size(); ++i)
			{
				int WordID = InputIDs[i];
				InputUser.insert(InputUser.end(), EmbeddingTable[WordID].begin(), EmbeddingTable[WordID].end());
			}
			*/

			Forward(InputUser);

			std::vector<float> ErrorOutput(OutputSize, 0.f);
			for (int i = 0; i < OutputSize; ++i)
			{
				ErrorOutput[i] = Predict[i] - Target[i];
				Biases2[i] -= (LearnRate * ErrorOutput[i]);
			}

			std::vector<float> ErrorHidden(HiddenSize, 0.f);
			for (int i = 0; i < HiddenSize; ++i)
			{
				for (int j = 0; j < OutputSize; ++j)
					ErrorHidden[i] += ErrorOutput[j] * MatrixWeight2[j][i];

				if (HiddenRaw[i] <= 0)
					ErrorHidden[i] = 0.f;
			}

			for (int i = 0; i < OutputSize; ++i)
				for (int j = 0; j < HiddenSize; ++j)
					MatrixWeight2[i][j] -= (LearnRate * ErrorOutput[i] * Hidden[j]);

			//std::vector<float> ErrorInput(InputSize, 0.f);
			for (int i = 0; i < HiddenSize; ++i)
			{
				Biases1[i] -= (LearnRate * ErrorHidden[i]);

				for (int j = 0; j < InputSize; ++j)
					MatrixWeight1[i][j] -= (LearnRate * ErrorHidden[i] * InputUser[j]);
					//ErrorInput[j] += ErrorHidden[i] * MatrixWeight1[i][j];
			}

			/*
			for (int i = 0; i < InputIDs.size(); ++i)
			{
				int ID = InputIDs[i];
				for (int j = 0; j < EmbeddingSize; ++j)
				{
					int OffsetIndex = (i * EmbeddingSize) + j;
					EmbeddingTable[ID][j] -= (LearnRate * ErrorInput[OffsetIndex]);
				}
			}
			*/
		}

		void SafeBrain(const std::string& FileName)
		{
			std::ofstream SafeFile(FileName, std::ios::binary);
			if (!SafeFile.is_open())
			{
				std::cout << "не удалось создать файл";
				return;
			}

			for (int i = 0; i < HiddenSize; ++i)
				SafeFile.write(reinterpret_cast<char*>(MatrixWeight1[i].data()), InputSize * sizeof(float));
			SafeFile.write(reinterpret_cast<char*>(Biases1.data()), HiddenSize * sizeof(float));

			for (int i = 0; i < OutputSize; ++i)
				SafeFile.write(reinterpret_cast<char*>(MatrixWeight2[i].data()), HiddenSize * sizeof(float));
			SafeFile.write(reinterpret_cast<char*>(Biases2.data()), OutputSize * sizeof(float));

			//for (int i = 0; i < OutputSize; ++i)
			//	SafeFile.write(reinterpret_cast<char*>(EmbeddingTable[i].data()), EmbeddingSize * sizeof(float));

			SafeFile.close();
			std::cout << "файл сохранен " << FileName;
		}

		bool LoadBrain(const std::string& FileName)
		{
			std::ifstream LoadFile(FileName, std::ios::binary);
			if (!LoadFile.is_open())
				return false;

			for (int i = 0; i < HiddenSize; ++i)
				LoadFile.read(reinterpret_cast<char*>(MatrixWeight1[i].data()), InputSize * sizeof(float));
			LoadFile.read(reinterpret_cast<char*>(Biases1.data()), HiddenSize * sizeof(float));

			for (int i = 0; i < OutputSize; ++i)
				LoadFile.read(reinterpret_cast<char*>(MatrixWeight2[i].data()), HiddenSize * sizeof(float));
			LoadFile.read(reinterpret_cast<char*>(Biases2.data()), OutputSize * sizeof(float));

			//for (int i = 0; i < OutputSize; ++i)
			//	LoadFile.read(reinterpret_cast<char*>(EmbeddingTable[i].data()), EmbeddingSize * sizeof(float));
		
			LoadFile.close();
			std::cout << "файл был успешно загружен из " << FileName;

			return true;
		}

	private:
		int InputSize = 0;
		int HiddenSize = 0;
		int EmbeddingSize = 0;
		int OutputSize = 0;
	};
}