import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.IntStream;

public class kNN2 {

    /**
     * @author Erdem Elik
     * @version 1.1
     */

    public static void main(String[] args) throws FileNotFoundException {
        // Data Parsed
        final List<List<Float>> trainData = new ArrayList<>(parseData(new File("train_data.txt")));
        final List<List<Float>> testData = new ArrayList<>(parseData(new File("test_data.txt")));

        // Label Parsed
        final List<Integer> trainLabel = new ArrayList<>(parseLabel(new File("train_label.txt")));
        final List<Integer> testLabel = new ArrayList<>(parseLabel(new File("test_label.txt")));

        List<String> chromosomeSet = generateInitialPopulation(5, testData.get(0).size());

        calculateGeneticAlgorithm(testData, trainData, testLabel, trainLabel, chromosomeSet, 98.0);

    }

    //////////////////////////////////////////////
    //////////////// DATA PARSING ////////////////
    //////////////////////////////////////////////

    /**
     * Parses file to float and creates a 2D List of features
     * 
     * @param file Train and test data to parse
     * @return List<List<Float>> parsed data
     */
    public static List<List<Float>> parseData(File file) throws FileNotFoundException {
        List<List<Float>> dataList = new ArrayList<>();
        Scanner dataScanner = new Scanner(file);
        while (dataScanner.hasNextLine()) {
            dataList.add(new ArrayList<>(Arrays.asList(
                    Arrays.stream(dataScanner.nextLine().split(" ")).map(Float::valueOf).toArray(Float[]::new))));
        }
        dataScanner.close();
        return dataList;
    }

    ///////////////////////////////////////////////
    //////////////// LABEL PARSING ////////////////
    ///////////////////////////////////////////////

    /**
     * Parses String labels to Integer and creates an List<Integer>
     * 
     * @param file File containing labels
     * @return A List of labels as Integer
     */
    public static List<Integer> parseLabel(File file) throws FileNotFoundException {
        List<Integer> labelArr = new ArrayList<>();
        Scanner labelReader = new Scanner(file);
        while (labelReader.hasNext()) {
            labelArr.add(Integer.parseInt(labelReader.next()));
        }
        labelReader.close();
        return labelArr;
    }

    ////////////////////////////////////////////////
    //////////////// END OF PARSING ////////////////
    ////////////////////////////////////////////////

    //////////////////////////////////////////////
    //////////////// CALCULATIONS ////////////////
    //////////////////////////////////////////////

    /**
     * Calculates the Euclidean distance of every test pattern to all training
     * patterns
     * 
     * @param testSet  Test Patterns
     * @param trainSet Train Patterns
     * @return 2D List of Euclidean distances
     */
    private static List<List<Float>> calculateEuclideanDistances(List<List<Float>> testInput,
            List<List<Float>> trainInput) {
        List<List<Float>> distances = new ArrayList<>();
        List<Float> innerArr;
        for (int i = 0; i < testInput.size(); i++) {
            innerArr = new ArrayList<>();
            for (int j = 0; j < trainInput.size(); j++) {
                double sum = 0;
                for (int k = 0; k < testInput.get(i).size(); k++) {
                    sum += Math.pow(testInput.get(i).get(k) - trainInput.get(j).get(k), 2);
                }
                innerArr.add((float) Math.sqrt(sum));
            }
            distances.add(innerArr);
        }
        return distances;
    }

    /**
     * Calculates the Manhattan distance of every test pattern to all training
     * patterns
     * 
     * @param testInput 2D List of test patterns
     * @param trainMap  Map of the train patterns
     * @return 2D List of Manhattan distances
     */
    private static List<List<Float>> calculateManhattanDistances(List<List<Float>> testInput, List<List<Float>> trainInput) {
        List<List<Float>> distances = new ArrayList<>();
        List<Float> innerArr;
        for (int i = 0; i < testInput.size(); i++) {
            innerArr = new ArrayList<>();
            for (int j = 0; j < trainInput.size(); j++) {
                double sum = 0;
                for (int k = 0; k < testInput.get(i).size(); k++) {
                    sum += Math.abs(testInput.get(i).get(k) - trainInput.get(j).get(k));
                }
                innerArr.add((float) sum);
            }
            distances.add(innerArr);
        }
        return distances;
    }

    /**
     * Compares predictions and actual labels to calculate the accuracy of the model
     * 
     * @param predictions Prediction labels
     * @param actuals     Test labels
     * @return Accuracy of the predictions
     */
    public static Double calculateAccuracy(List<Integer> predictions, List<Integer> actuals) {
        try {
            return (double) IntStream.range(0, actuals.size()).filter(i -> actuals.get(i).equals(predictions.get(i)))
                    .count() / actuals.size() * 100;
        } catch (IndexOutOfBoundsException e) {
            return null;
        }
    }

    /////////////////////////////////////////////////////
    //////////////// END OF CALCULATIONS ////////////////
    /////////////////////////////////////////////////////

    
    /////////////////////////////////////////////////////////////
    //////////////// PREDICTIONS & FILE PRINTING ////////////////
    /////////////////////////////////////////////////////////////

    /**
     * Finds the label of the nearest neighbour
     * 
     * @param inputToPredict List containing distances
     * @param trainLabels    List of TRAINING labels
     * @return List of predictions
     */
    public static List<Integer> predictLabel(List<List<Float>> inputToPredict, List<Integer> trainLabels) {
        List<Integer> predictions = new ArrayList<>();
        for (List<Float> innerArr : inputToPredict) {
            int minIndex = trainLabels.get(innerArr.indexOf(Collections.min(innerArr)));
            predictions.add(minIndex);
        }
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("output2.txt"))) {
            writer.write(predictions.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        return predictions;

    }

    //////////////////////////////////////////////////////////
    //////////////// BINARY GENETIC ALGORITHM ////////////////
    //////////////////////////////////////////////////////////

    public static Double shortcutEuclidean(List<List<Float>> testData, List<List<Float>> trainData, List<Integer> testLabel,
            List<Integer> trainLabel) {
        return calculateAccuracy(predictLabel(calculateEuclideanDistances(testData, trainData), trainLabel), testLabel);
    }
    public static Double shortcutManhattan(List<List<Float>> testData, List<List<Float>> trainData, List<Integer> testLabel,
            List<Integer> trainLabel) {
        return calculateAccuracy(predictLabel(calculateManhattanDistances(testData, trainData), trainLabel), testLabel);
    }

    /**
     * Generates n-number of random binary strings of length 61 to be used as
     * initial population
     * 
     * @param initialPopulation starting population of the genetic algorithm
     * @return List of String of length initialPopulation containing binary strings of
     *         length n
     */
    public static List<String> generateInitialPopulation(int initialPopulation, int length) {
        List<String> binaryStrings = new ArrayList<>();
        Random random = new Random();

        for (int i = 0; i < initialPopulation; i++) {
            StringBuilder binaryString = new StringBuilder();

            for (int j = 0; j < length; j++) {
                int randomBit = random.nextInt(2);
                binaryString.append(randomBit);
            }
            binaryStrings.add(binaryString.toString());
        }

        return binaryStrings;
    }

    /**
     * Finds and returns the indices of 1's in param chromosomeSet
     * 
     * @param chromosomeSet Contains arbitrary number of binary strings of arbitrary length
     * @return 2D List of Integers with inner List containing the indices of 1's per binary string
     */
    public static List<List<Integer>> findIndicesofOnes(List<String> chromosomeSet) {

        List<List<Integer>> indices = new ArrayList<>();

        for (String binaryString : chromosomeSet) {
            List<Integer> innerIndices = new ArrayList<>();
            for (int i = 0; i < binaryString.length(); i++) {
                if (binaryString.charAt(i) == '1') {
                    innerIndices.add(i);
                }
            }
            indices.add(innerIndices);
        }

        return indices;
    }

    /**
     * Use this after with findIndicesOfOnes to get the feature columns indicated the indices returned by findIndicesOfOnes
     * 
     * @param indices List of indices, which will be used to retrieve the suggested columns of data from param dataSet
     * @param dataSet preferably trainData and testData
     * @return 2D List of Floats representing the new trainData and testData consisting of columns specified by the indices in param indices
     */
    public static List<List<Float>> retrieveDataFromBinaryString(List<Integer> indices, List<List<Float>> dataSet) {
        List<List<Float>> retrievedPattern = new ArrayList<>();
        for (int i = 0; i < dataSet.size(); i++) {
            List<Float> retrievedDataPoint = new ArrayList<>();
            for (int indexToLookUp : indices) { // Fix variable name from 'list' to 'indices'
                retrievedDataPoint.add(dataSet.get(i).get(indexToLookUp));
            }
            retrievedPattern.add(retrievedDataPoint);
        }
        return retrievedPattern;
    }

    /**
     * Initially, feeds itself with the chromosomeSet. Later on, calls chromosomeMutator() to randomly mutate the fittest chromosome
     * and feed itself with the mutated chromosome. Loops this until param accuracyThreshold has been met or passed
     * 
     * @param testSet 2D List of test patterns
     * @param trainSet 2D List of train patterns
     * @param testLabel List of Integers labeling testSet
     * @param trainLabel List of Integers labeling trainSet
     * @param chromosomeSet List of Strings containing binary string produced by generateInitalPopulation(), has arbitrary size and length
     * @param accuracyThreshold Threshold to meet while iterating
     */
    public static void calculateGeneticAlgorithm(List<List<Float>> testSet, List<List<Float>> trainSet, List<Integer> testLabel, List<Integer> trainLabel, List<String> chromosomeSet, double accuracyThreshold) {
        final List<List<Float>> localTestSet = new ArrayList<>(testSet);
        final List<List<Float>> localTrainSet = new ArrayList<>(trainSet);
        NumberFormat formatter = new DecimalFormat("###.0");

        List<String> newGen = new ArrayList<>(chromosomeSet);

        List<List<Integer>> indices = new ArrayList<>(findIndicesofOnes(chromosomeSet));

        
        double generationSuccessPercentage = 0;
        String bestChromosome = "";
        List<Double> listOfResults = new ArrayList<>();

        while(generationSuccessPercentage < accuracyThreshold) {
            listOfResults =  new ArrayList<>();
            List<List<Float>> modifiedTestSet; 
            List<List<Float>> modifiedTrainSet;
        
            for (int i = 0; i < indices.size(); i++) {
                modifiedTestSet = retrieveDataFromBinaryString(indices.get(i), localTestSet); // Ready for distance measuring
                modifiedTrainSet = retrieveDataFromBinaryString(indices.get(i), localTrainSet); // Ready for distance measuring
        
                Double result = shortcutManhattan(modifiedTestSet, modifiedTrainSet, testLabel, trainLabel);
                listOfResults.add(result);
            }
        
            int bestIndex = listOfResults.indexOf(Collections.max(listOfResults));
            bestChromosome = newGen.get(bestIndex);
        
            newGen = chromosomeMutator(bestChromosome, 4, 2);
            indices = findIndicesofOnes(newGen);
        
            // Move these lines inside the loop after the loop body
            modifiedTestSet = retrieveDataFromBinaryString(indices.get(bestIndex), localTestSet);
            modifiedTrainSet = retrieveDataFromBinaryString(indices.get(bestIndex), localTrainSet);
            generationSuccessPercentage = shortcutManhattan(modifiedTestSet, modifiedTrainSet, testLabel, trainLabel);
            
            generationSuccessPercentage = Collections.max(listOfResults);
        }
        System.out.println("Best is " + bestChromosome + " with " + formatter.format(Collections.max(listOfResults)) + "%");
        // return generationSuccessPercentage;
    }
        

    /**
     * Mutates param chromosome randomly and returns it
     * @param chromosome Preferably fittest binary chromosome with arbitrary length
     * @param offSpringCount Number of offsprings to produce
     * @param mutationCount Number of mutation apply to chromosome
     * @return a list of size param offSpringCount containing new chromosomes (binary strings)
     */
    public static List<String> chromosomeMutator(String chromosome, int offSpringCount, int mutationCount) {

        List<String> nextGenerationChromosomes = new ArrayList<>();
        Random rand = new Random();
        
        for (int i = 0; i < offSpringCount; i++) {
            String c1 = chromosome;
            
            for (int j = 0; j < mutationCount; j++) {
                
                int length = chromosome.length();
                int crossoverPointStart = rand.nextInt(5, length / 2);
                int crossoverPointEnd = rand.nextInt(crossoverPointStart,length);
                String c2 = generateInitialPopulation(1, crossoverPointEnd - crossoverPointStart).get(0);
                c1 = c1.substring(0, crossoverPointStart) + c2 + c1.substring(crossoverPointEnd);
            }
                
            nextGenerationChromosomes.add(c1);
        }

        return nextGenerationChromosomes;
    }
    
}
