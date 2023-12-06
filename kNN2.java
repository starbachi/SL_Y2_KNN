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
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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

        // Data Parsed
        final List<List<Float>> trainSet = new ArrayList<>(parseData(new File("train_data.txt")));
        final List<List<Float>> testSet = new ArrayList<>(parseData(new File("test_data.txt")));

        // Label Parsed
        final List<Integer> trainLabel = new ArrayList<>(parseLabel(new File("train_label.txt")));
        final List<Integer> testLabel = new ArrayList<>(parseLabel(new File("test_label.txt")));

        calculateGeneticAlgorithm(testSet, trainSet, testLabel, trainLabel, 90, 200, testSet.get(0).size(), 30);

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
    public static void calculateGeneticAlgorithm(List<List<Float>> testSet, List<List<Float>> trainSet,
            List<Integer> testLabel,
            List<Integer> trainLabel, double accuracyThreshold, int initialPopulationSize, int initalPopulationLength,
            int mutationChance) {

        final List<List<Float>> localTestSet = new ArrayList<>(testSet);
        final List<List<Float>> localTrainSet = new ArrayList<>(trainSet);
        NumberFormat formatter = new DecimalFormat("###.0");

        // Initally populated via params, later re-initated with new generations
        List<String> parentSet = generateinitialPopulation(initialPopulationSize, initalPopulationLength);

        // Initally populated via params, later re-initated with new generations

        Map<String, Double> accuracyMap = new HashMap<>();

        // Map.Entry<String, Double> entryAtIndex0 =
        // accuracyMap.entrySet().stream().findFirst().orElse(null);
        // String keyAtIndex0 = entryAtIndex0.getKey();
        // Double valueAtIndex0 = entryAtIndex0.getValue();

        List<Double> accuracies = new ArrayList<>();
        int count = 0;
        List<String> mutatedParentSet = new ArrayList<>(parentSet);

        while (accuracies.isEmpty() || accuracies.get(accuracies.size() - 1) < accuracyThreshold) {
            accuracies.clear();
            mutatedParentSet = crossover(mutatedParentSet, mutationChance);
            List<List<Integer>> indices = new ArrayList<>(findIndicesofOnes(mutatedParentSet));
            for (int i = 0; i < indices.size(); i++) {
                List<List<Float>> modifiedTrainList = new ArrayList<>(
                        retrieveDataFromBinaryString(indices.get(i), localTrainSet));
                List<List<Float>> modifiedTestList = new ArrayList<>(
                        retrieveDataFromBinaryString(indices.get(i), localTestSet));

                double result = shortcutEuclidean(modifiedTestList, modifiedTrainList, testLabel, trainLabel);
                accuracyMap.put((mutatedParentSet).get(i), result);
                accuracies.add(result);
                // System.out.println(mutatedParentSet.get(i) + " " +
                // shortcutEuclidean(modifiedTestList, modifiedTrainList, testLabel,
                // trainLabel));
            }
            //TODO you are not iterating over the best 50 of accuracyMap FIX!!!
            accuracyMap = cutPopInHalf(accuracyMap);
            accuracies.sort(Comparator.naturalOrder());
            // System.out.println(accuracies.toString());
            count++;
            // System.out.println("Generation: " + count + " best: "+
            // accuracies.get(accuracies.size()) );
            // System.out.println(accuracies.toString());
            System.out.println(accuracies.get(accuracies.size() - 1));

            accuracies = accuracies.subList(accuracies.size() / 2, accuracies.size());
        }
        System.out.println(getKeysByValue(accuracyMap, accuracies.get(accuracies.size() - 1)) + " with accuracy "
                + accuracies.get(accuracies.size() - 1));
    }

    public static <K, V> List<K> getKeysByValue(Map<K, V> map, V value) {
        List<K> keys = new ArrayList<>();

        for (Map.Entry<K, V> entry : map.entrySet()) {
            if (Objects.equals(value, entry.getValue())) {
                keys.add(entry.getKey());
            }
        }

        return keys;
    }

    public static List<String> crossover(List<String> parentSet, int mutationChance) {

        List<String> xOverDone = new ArrayList<>();
        Random rand = new Random();
        while (!parentSet.isEmpty()) {
            String p1 = parentSet.get(0);
            parentSet.remove(0);
            String p2 = parentSet.get(0);
            parentSet.remove(0);

            int crossoverStart = rand.nextInt(0, Math.round(p1.length() / 2));
            int crossoverEnd = rand.nextInt(crossoverStart, p1.length());

            String newP1 = p1.substring(0, crossoverStart) + p2.substring(crossoverStart, crossoverEnd)
                    + p1.substring(crossoverEnd);
            String newP2 = p2.substring(0, crossoverStart) + p1.substring(crossoverStart, crossoverEnd)
                    + p2.substring(crossoverEnd);

            
            xOverDone = new ArrayList<>(parentMutator(xOverDone = new ArrayList<>(List.of(newP1, newP2)), mutationChance));
        }
        return xOverDone;
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
    public static List<String> parentMutator(List<String> parentSet, int mutationChance) {

        String p1 = parentSet.get(0);
        String p2 = parentSet.get(1);

        List<String> nextGenIncludedSet = new ArrayList<>();
        Random rand = new Random();

        StringBuilder sbp1 = new StringBuilder(p1);
        StringBuilder sbp2 = new StringBuilder(p2);

        for (int i = 0; i < p1.length(); i++) {
            if (rand.nextInt(100) < mutationChance) {
                sbp1.setCharAt(i, (p1.charAt(i) == '0') ? '1' : '0');
            }
        }

        for (int i = 0; i < p2.length(); i++) {
            if (rand.nextInt(100) < mutationChance) {
                sbp2.setCharAt(i, (p2.charAt(i) == '0') ? '1' : '0');
            }
        }

        p1 = sbp1.toString();
        p2 = sbp2.toString();

        nextGenIncludedSet = new ArrayList<>(List.of(p1, p2));
        return nextGenIncludedSet;
    }


    public static void populationIncrement(List<String> population) {
    }

    public static Map<String, Double> cutPopInHalf(Map<String, Double> populationMap) {
        Map<String, Double> sortedAccuracyMap = populationMap.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue())
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1,
                        LinkedHashMap::new));

        int mapSizeHalved = sortedAccuracyMap.size() / 2;

        while (sortedAccuracyMap.size() > mapSizeHalved) {
            String lastKey = sortedAccuracyMap.keySet().iterator().next();

            sortedAccuracyMap.remove(lastKey);
        }

        return sortedAccuracyMap;
    }
}