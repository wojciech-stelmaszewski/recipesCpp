#include<iostream>
#include<fstream>
#include<chrono>
#include<algorithm>
#include<ctime>
#include<random>
#include<Eigen/Sparse>
#include<Eigen/Dense>

#define MinimumIngredientsCount 5
#define MaximumIngredientsCount 20

#define MinimumIngredientValue 1
#define MaximumIngredientValue 999

#define OutputFileName "output.txt"

std::random_device randomDevice;
//std::mt19937 generator(randomDevice());
std::mt19937 generator(0);
std::uniform_int_distribution<> randomIngredientsCountGenerator(MinimumIngredientsCount, MaximumIngredientsCount);
std::uniform_int_distribution<> randomIngredientValueGenerator(MinimumIngredientValue, MaximumIngredientValue);

long getIngredientsCount(long);
float getIngredientValue();
Eigen::SparseVector<float> createRecipe(long);
std::vector<Eigen::Triplet<float>> createRecipes(long, long, long*);
void saveMatrixToFile(Eigen::SparseMatrix<float>);

int main(int argc, char** argv)
{
	long recipesCount = 10000000L;
	long recipeDimension = 5000L;

	std::cout << "Generowanie danych testowych!" << std::endl;

	long elementsCount;
	auto triplets = createRecipes(recipeDimension, recipesCount, &elementsCount);

	Eigen::SparseMatrix<float> recipesMatrix(recipeDimension, recipesCount);
	recipesMatrix.setFromTriplets(triplets.begin(), triplets.end());

	std::cout << "Dane testowe WYGENEROWANE!" << std::endl;
	std::cout << "Liczba przepisow: " << recipesCount << std::endl;
	std::cout << "Liczba wymiarow przestrzeni przypisow: " << recipeDimension << std::endl;
	std::cout << "Liczba niezerowych elementow w rzadkiej macierzy przepisow: " << elementsCount << std::endl;

	std::cout << std::endl << "Rozpoczynamy OBLICZENIA!" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();

	auto resultMatrix = (recipesMatrix.transpose()*recipesMatrix).pruned();

	auto finish = std::chrono::high_resolution_clock::now();

	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
	std::cout << std::endl << "Obliczenia ZAKONCZONE! Trwaly: " << milliseconds.count() / 1000.0f << "s" << std::endl;
	
	//std::cout << "Rezultat obliczen zostanie zapisany do pliku: " << OutputFileName << std::endl;
	//saveMatrixToFile(resultMatrix);

	std::cout << "KONIEC. Nacisnij dowolny przycisk na klawiaturze." << std::endl;
	std::getchar();

	return 0;
}

void saveMatrixToFile(Eigen::SparseMatrix<float> matrix)
{
	std::ofstream outputFile(OutputFileName);
	if (outputFile.is_open())
	{
		outputFile << matrix;
	}

	outputFile.close();
}

std::vector<Eigen::Triplet<float>> createRecipes(long dimension, long recipesCount, long* tripletsCount)
{
	std::vector<Eigen::Triplet<float>> recipesTriplets;

	*tripletsCount = 0L;

	for (int i = 0; i < recipesCount; i++)
	{
		Eigen::SparseVector<float> recipe = createRecipe(dimension);

		Eigen::SparseVector<float>::InnerIterator recipeNonZeroIterator(recipe);
		for (;recipeNonZeroIterator;++recipeNonZeroIterator)
		{
			recipesTriplets.push_back(Eigen::Triplet<float>(recipeNonZeroIterator.index(), i, recipeNonZeroIterator.value()));
		}

		*tripletsCount += recipe.nonZeros();
	}

	return recipesTriplets;
}

long getIngredientsCount(long dimension)
{
	return std::min(dimension, (long)randomIngredientsCountGenerator(generator));
}

float getIngredientValue()
{
	return randomIngredientValueGenerator(generator) / 10.0f;
}

Eigen::SparseVector<float> createRecipe(long dimension)
{
	std::uniform_int_distribution<> randomIndicesGenerator(0, dimension - 1);

	Eigen::SparseVector<float> recipe(dimension);

	int ingredientsCount = getIngredientsCount(dimension);

	std::vector<int> indices(ingredientsCount);
	std::fill(indices.begin(), indices.end(), -1);
	int counter = 0;

	while (counter < ingredientsCount) 
	{
		int randomIndex = randomIndicesGenerator(generator);
		if (std::find(indices.begin(), indices.end(), randomIndex) == indices.end())
		{
			indices[counter++] = randomIndex;
			recipe.coeffRef(randomIndex) = getIngredientValue();
		}
	}

	return recipe / recipe.norm();
}