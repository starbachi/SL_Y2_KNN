import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.security.KeyStore.Entry;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class kNN2 {

    /**
     * @author Erdem Elik
     * @version 1.1
     */

    public static void main(String[] args) throws FileNotFoundException {
        final List<List<Float>> trainSet = new ArrayList<>(parseData(new File("train_data.txt")));

        // Data Parsed
        final List<List<Float>> testSet = new ArrayList<>(parseData(new File("test_data.txt")));

        // Label Parsed
        final List<Integer> trainLabel = new ArrayList<>(parseLabel(new File("train_label.txt")));
        final List<Integer> testLabel = new ArrayList<>(parseLabel(new File("test_label.txt")));

        calculateGeneticAlgorithm(99.5, 200, 5, testSet, trainSet, testLabel, trainLabel, testSet.get(0).size());

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
    private static List<List<Float>> calculateManhattanDistances(List<List<Float>> testInput,
            List<List<Float>> trainInput) {
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

    public static Double shortcutEuclidean(List<List<Float>> testSet, List<List<Float>> trainSet,
            List<Integer> testLabel,

            List<Integer> trainLabel) {
        return calculateAccuracy(predictLabel(calculateEuclideanDistances(testSet, trainSet), trainLabel), testLabel);
    }

    public static Double shortcutManhattan(List<List<Float>> testData, List<List<Float>> trainData,
            List<Integer> testLabel,
            List<Integer> trainLabel) {
        return calculateAccuracy(predictLabel(calculateManhattanDistances(testData, trainData), trainLabel), testLabel);
    }

    //////////////////////////////////////////////////////////
    //////////////// BINARY GENETIC ALGORITHM ////////////////
    //////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////
    //////////////// INITIAL POPULATION GENERATION ////////////////
    ///////////////////////////////////////////////////////////////

    /**
     * Generates n-number of random binary strings of length 61 to be used as
     * initial population
     * 
     * @param initialPopulationSize starting population of the genetic algorithm
     * @return List of String of length initialPopulationSize containing binary
     *         strings of
     *         length n
     */
    public static List<String> generateinitialPopulation(int initialPopulationSize, int length) {
        List<String> binaryStrings = new ArrayList<>();
        Random random = new Random();

        for (int i = 0; i < initialPopulationSize; i++) {
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
     * @param parentSet Contains arbitrary number of binary strings of arbitrary
     *                  length
     * @return 2D List of Integers with inner List containing the indices of 1's per
     *         binary string
     */
    public static List<List<Integer>> findIndicesofOnes(List<String> parentSet) {

        List<List<Integer>> indices = new ArrayList<>();

        for (String binaryString : parentSet) {
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
     * Use this after with findIndicesOfOnes to get the feature columns indicated
     * the indices returned by findIndicesOfOnes
     * 
     * @param indices List of indices, which will be used to retrieve the suggested
     *                columns of data from param dataSet
     * @param dataSet preferably trainData and testData
     * @return 2D List of Floats representing the new trainData and testData
     *         consisting of columns specified by the indices in param indices
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
     * Initially, feeds itself with the parentSet. Later on, calls parentMutator()
     * to randomly mutate the fittest parent
     * and feed itself with the mutated parent. Loops this until param
     * accuracyThreshold has been met or passed
     * 
     * @param testSet           2D List of test patterns
     * @param trainSet          2D List of train patterns
     * @param testLabel         List of Integers labeling testSet
     * @param trainLabel        List of Integers labeling trainSet
     * @param parentSet         List of Strings containing parents produced by
     *                          generateInitalPopulation(), has arbitrary size and
     *                          length
     * @param accuracyThreshold Threshold to meet while iterating
     */
    public static void calculateGeneticAlgorithm(double accuracyThreshold, int initialPopulationSize,
            int mutationChance, List<List<Float>> testSet, List<List<Float>> trainSet, List<Integer> testLabel,
            List<Integer> trainLabel, int initalPopulationLength) {

        final List<List<Float>> localTestSet = new ArrayList<>(testSet);
        final List<List<Float>> localTrainSet = new ArrayList<>(trainSet);
        NumberFormat formatter = new DecimalFormat("###.0");

        List<String> parentSet = generateinitialPopulation(initialPopulationSize, initalPopulationLength);

        List<String> mutatedParentSet = new ArrayList<>(parentSet);

        String elite = "";

        // STORES THE ACCURACY OF EVERY PARENT
        List<Double> resultSet = new ArrayList<>();

        while (resultSet.isEmpty() || resultSet.get(0) < accuracyThreshold) {

            resultSet.clear();
            Map<String, Double> accuracyMap = new HashMap<>();

            // APPLY CROSSOVER
            List<String> storeMutations = new ArrayList<>(crossover(mutatedParentSet));

            // APPLY MUTATION
            storeMutations = mutator(storeMutations, mutationChance);

            // STORE BOTH OLD GENERATION AND NEW GENERATION (2X INITIAL POPULATION SIZE)
            for (int i = 0; i < storeMutations.size(); i++) {
                mutatedParentSet.add(storeMutations.get(i));
            }

            // GET THE INDICES OF MUTATED PARENTS
            List<List<Integer>> indices = new ArrayList<>(findIndicesofOnes(mutatedParentSet));

            for (int i = 0; i < indices.size(); i++) {

                // GET THE DATA AT THE INDICES
                List<List<Float>> modifiedTrainSet = new ArrayList<>(
                        retrieveDataFromBinaryString(indices.get(i), localTrainSet));
                List<List<Float>> modifiedTestSet = new ArrayList<>(
                        retrieveDataFromBinaryString(indices.get(i), localTestSet));

                double result = shortcutEuclidean(modifiedTestSet, modifiedTrainSet, testLabel, trainLabel);

                accuracyMap.put(mutatedParentSet.get(i), result);
                resultSet.add(result);
            }

            mutatedParentSet.clear();
            resultSet.sort(Comparator.reverseOrder());
            accuracyMap = accuracyMap.entrySet()
                    .stream()
                    .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                    .collect(Collectors.toMap(
                            Map.Entry::getKey,
                            Map.Entry::getValue,
                            (oldValue, newValue) -> oldValue, LinkedHashMap::new));
            List<String> mostAccurateParents = new ArrayList<>(accuracyMap.keySet());
            // elite1 = mostAccurateParents.get(0);
            // elite2 = mostAccurateParents.get(1);

            mutatedParentSet = mostAccurateParents.subList(0, mostAccurateParents.size() / 2);
            elite = mutatedParentSet.get(0);
            
            System.out.println(resultSet.get(0));
            System.out.println("Elite: " + elite);
        }
        System.out.println("Found best accuracy: " + resultSet.get(0) + " with parent " + mutatedParentSet.get(0));

    }

    public static List<String> crossover(List<String> parentSet) {

        List<String> localParentSet = new ArrayList<>(parentSet);
        List<String> returnList = new ArrayList<>();

        Random rand = new Random();

        returnList.add(localParentSet.get(0));
        returnList.add(localParentSet.get(1));

        for (int i = 2; i < localParentSet.size() - 1; i += 2) {
            int length = localParentSet.get(i).length();
            String p1 = localParentSet.get(0);
            String p2 = localParentSet.get(i + 1);

            int r1 = rand.nextInt(length);
            int r2 = rand.nextInt(length);

            int crossoverStart = (r1 < r2) ? r1 : r2;
            int crossoverEnd = (r1 < r2) ? r2 : r1;

            String swap = p1.substring(crossoverStart, crossoverEnd);
            p1 = p1.substring(0, crossoverStart) + p2.substring(crossoverStart, crossoverEnd)
                    + p1.substring(crossoverEnd);
            p2 = p2.substring(0, crossoverStart) + swap + p2.substring(crossoverEnd);

            returnList.add(p1);
            returnList.add(p2);
        }
        return returnList;
    }

    /**
     * Mutates parents in param parentMap randomly and returns it
     * 
     * @param parent         Preferably fittest binary chromosome with arbitrary
     *                       length
     * @param offSpringCount Number of offsprings to produce
     * @param mutationChance Chance of mutation, changes per chromosome
     * @return a list of size param offSpringCount containing new chromosomes
     *         (binary strings)
     */
    public static List<String> mutator(List<String> parentSet, int mutationChance) {

        List<String> modifiableParentSet = new ArrayList<>(parentSet);
        List<String> mutationDone = new ArrayList<>();

        mutationDone.add(modifiableParentSet.get(0));

        Random rand = new Random();

        for (int i = 1; i < modifiableParentSet.size() - 1; i += 2) {

            StringBuilder p1 = new StringBuilder(modifiableParentSet.get(0));

            StringBuilder p2 = new StringBuilder(modifiableParentSet.get(i + 1));

            for (int j = 0; j < p1.length(); j++) {
                if (rand.nextInt(100) < mutationChance) {
                    p1.setCharAt(j, (p1.charAt(j) == '0') ? '1' : '0');
                }
            }

            for (int j = 0; j < p2.length(); j++) {
                if (rand.nextInt(100) < mutationChance) {
                    p2.setCharAt(j, (p2.charAt(j) == '0') ? '1' : '0');
                }
            }

            mutationDone.add(p1.toString());
            mutationDone.add(p2.toString());
        }

        return mutationDone;
    }

}