#pragma once

#include <vector>
#include <numeric>
#include <cstdlib>
#include <fstream>

namespace Models
{
	class DeepModel
	{
	public:
		std::vector<std::vector<float>> matrixWeight1;
		std::vector<float> biases1;

		std::vector<std::vector<float>> matrixWeight2;
		std::vector<float> biases2;

		std::vector<float> hiddenRaw;
		std::vector<float> hidden;
		std::vector<float> predict;

		//std::vector<std::vector<float>> embeddingTable;

		float learnRate = 0.05f;
		
		DeepModel() = default;
		~DeepModel() = default;

		void init(int inSize, int hidSize, int outSize)
		{
			inputSize = inSize;
			hiddenSize = hidSize;
			outputSize = outSize;

			biases1.resize(hiddenSize, 0.f);
			biases2.resize(outputSize, 0.f);

			matrixWeight1.resize(hiddenSize, std::vector<float>(inputSize));
			matrixWeight2.resize(outputSize, std::vector<float>(hiddenSize));

			for (int i = 0; i < hiddenSize; ++i)
				for (int j = 0; j < inputSize; ++j)
					matrixWeight1[i][j] = (float(rand() % 100) / 1000.f) - 0.05f;

			for (int i = 0; i < outputSize; ++i)
				for (int j = 0; j < hiddenSize; ++j)
					matrixWeight2[i][j] = (float(rand() % 100) / 1000.f) - 0.05f;

			hiddenRaw.resize(hiddenSize, 0.f);
			hidden.resize(hiddenSize, 0.f);
			predict.resize(outputSize, 0.f);

			/*
			EmbeddingTable.resize(OutputSize, std::vector<float>(EmbeddingSize));

			for (int i = 0; i < OutputSize; ++i)
				for (int j = 0; j < EmbeddingSize; ++j)
					EmbeddingTable[i][j] = (float(rand() % 100) / 1000.f) - 0.05f;
			*/
		}

		std::vector<float> forward(const std::vector<float>& inputUser)
		{
			for (int i = 0; i < hiddenSize; ++i)
			{
				hiddenRaw[i] = biases1[i];
				for (int j = 0; j < inputSize; ++j)
					hiddenRaw[i] += matrixWeight1[i][j] * inputUser[j];

				hidden[i] = (hiddenRaw[i] < 0 ? 0.f : hiddenRaw[i]);
			}

			for (int i = 0; i < outputSize; ++i)
			{
				predict[i] = biases2[i];
				for (int j = 0; j < hiddenSize; ++j)
					predict[i] += matrixWeight2[i][j] * hidden[j];
			}

			return predict;
		}

		void machineLearning(const std::vector<float>& InputUser, const std::vector<float>& target)
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

			forward(InputUser);

			std::vector<float> errorOutput(outputSize, 0.f);
			for (int i = 0; i < outputSize; ++i)
			{
				errorOutput[i] = predict[i] - target[i];
				biases2[i] -= (learnRate * errorOutput[i]);
			}

			std::vector<float> errorHidden(hiddenSize, 0.f);
			for (int i = 0; i < hiddenSize; ++i)
			{
				for (int j = 0; j < outputSize; ++j)
					errorHidden[i] += errorOutput[j] * matrixWeight2[j][i];

				if (hiddenRaw[i] <= 0)
					errorHidden[i] = 0.f;
			}

			for (int i = 0; i < outputSize; ++i)
				for (int j = 0; j < hiddenSize; ++j)
					matrixWeight2[i][j] -= (learnRate * errorOutput[i] * hidden[j]);

			//std::vector<float> ErrorInput(InputSize, 0.f);
			for (int i = 0; i < hiddenSize; ++i)
			{
				biases1[i] -= (learnRate * errorHidden[i]);

				for (int j = 0; j < inputSize; ++j)
					matrixWeight1[i][j] -= (learnRate * errorHidden[i] * InputUser[j]);
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

		void safeBrain(const std::string& fileName)
		{
			std::ofstream safeFile(fileName, std::ios::binary);
			if (!safeFile.is_open())
			{
				//std::cout << "не удалось создать файл";
				return;
			}

			for (int i = 0; i < hiddenSize; ++i)
				safeFile.write(reinterpret_cast<char*>(matrixWeight1[i].data()), inputSize * sizeof(float));
			safeFile.write(reinterpret_cast<char*>(biases1.data()), hiddenSize * sizeof(float));

			for (int i = 0; i < outputSize; ++i)
				safeFile.write(reinterpret_cast<char*>(matrixWeight2[i].data()), hiddenSize * sizeof(float));
			safeFile.write(reinterpret_cast<char*>(biases2.data()), outputSize * sizeof(float));

			//for (int i = 0; i < OutputSize; ++i)
			//	SafeFile.write(reinterpret_cast<char*>(EmbeddingTable[i].data()), EmbeddingSize * sizeof(float));

			safeFile.close();
			//std::cout << "файл сохранен " << FileName;
		}

		bool loadBrain(const std::string& fileName)
		{
			std::ifstream loadFile(fileName, std::ios::binary);
			if (!loadFile.is_open())
				return false;

			for (int i = 0; i < hiddenSize; ++i)
				loadFile.read(reinterpret_cast<char*>(matrixWeight1[i].data()), inputSize * sizeof(float));
			loadFile.read(reinterpret_cast<char*>(biases1.data()), hiddenSize * sizeof(float));

			for (int i = 0; i < outputSize; ++i)
				loadFile.read(reinterpret_cast<char*>(matrixWeight2[i].data()), hiddenSize * sizeof(float));
			loadFile.read(reinterpret_cast<char*>(biases2.data()), outputSize * sizeof(float));

			//for (int i = 0; i < OutputSize; ++i)
			//	LoadFile.read(reinterpret_cast<char*>(EmbeddingTable[i].data()), EmbeddingSize * sizeof(float));
		
			loadFile.close();
			//std::cout << "файл был успешно загружен из " << FileName;

			return true;
		}

	private:
		int inputSize = 0;
		int hiddenSize = 0;
		int embeddingSize = 0;
		int outputSize = 0;
	};
}