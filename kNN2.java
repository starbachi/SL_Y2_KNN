import java.io.BufferedWriter;
import java.io.File;
import java.nio.file.*;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class kNN2 {

    /**
     * @author Erdem Elik
     * @version 1.1
     */
    static final String TRAIN_DATA_PATH = "train_data.txt";
    static final String TEST_DATA_PATH = "test_data.txt";
    static final String TRAIN_LABEL_PATH = "train_label.txt";
    static final String TEST_LABEL_PATH = "test_label.txt";
    static List<List<Float>> TRAIN_DATA = new ArrayList<>();
    static List<List<Float>> TEST_DATA = new ArrayList<>();
    static List<Integer> TEST_LABEL = new ArrayList<>();
    static List<Integer> TRAIN_LABEL = new ArrayList<>();
    static int initalPopulationLength;

    public static void main(String[] args) {
        try {
            TRAIN_DATA = parseData(Paths.get(TRAIN_DATA_PATH).toFile());
            TEST_DATA = parseData(Paths.get(TEST_DATA_PATH).toFile());
            TRAIN_LABEL = parseLabel(Paths.get(TRAIN_LABEL_PATH).toFile());
            TEST_LABEL = parseLabel(Paths.get(TEST_LABEL_PATH).toFile());
            initalPopulationLength = TRAIN_DATA.get(0).size();

            String a = calculateGeneticAlgorithm(100, 500, 5);
            System.out.println(
                    "Solo test accuracy is " + soloTest(TEST_DATA, TRAIN_DATA, TEST_LABEL, TRAIN_LABEL, a) + " for "
                            + a);
        } catch (FileNotFoundException e) {
            System.out.println("One or more files were not found.");
            e.printStackTrace();
        }
    }

    /**
     * Verifies the accuracy of the fittest parent from the genetic algorithm.
     *
     * @param testSet    The test set, represented as a list of points. Each point
     *                   is a list of floats.
     * @param trainSet   The training set, represented as a list of points. Each
     *                   point is a list of floats.
     * @param testLabel  The labels for the test set.
     * @param trainLabel The labels for the training set.
     * @param parent     A binary string representing which features to use for the
     *                   test. A '1' at a position indicates that the corresponding
     *                   feature should be used, and a '0' indicates that it should
     *                   be ignored.
     * @return The accuracy of the predictions, represented as a double.
     */
    public static Double soloTest(List<List<Float>> testSet, List<List<Float>> trainSet, List<Integer> testLabel,
            List<Integer> trainLabel, String parent) {
        List<List<Float>> testList = retrieveDataFromBinaryString(findIndicesofOnes(List.of(parent)).get(0), testSet);
        List<List<Float>> trainList = retrieveDataFromBinaryString(findIndicesofOnes(List.of(parent)).get(0), trainSet);
        List<List<Float>> eucs = calculateEuclideanDistances(testList, trainList);
        List<Integer> pred = predictLabel(eucs, trainLabel);

        return calculateAccuracy(pred, testLabel);
    }

    //////////////////////////////////////////////
    //////////////// DATA PARSING ////////////////
    //////////////////////////////////////////////

    /**
     * This method reads a file containing data points, parses the points from
     * String to Float, and stores them in a 2D list.
     * Each line in the file should represent a data point, with the features of the
     * point separated by spaces.
     *
     * @param file The file containing data points.
     * @return A 2D list of data points, where each inner list represents a data
     *         point and contains the features of the point as Floats.
     * @throws FileNotFoundException If the provided file does not exist.
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
     * This method reads a file containing labels, parses the labels from String to
     * Integer, and stores them in a list.
     *
     * @param file The file containing labels. Each label should be represented as a
     *             separate String.
     * @return A list of labels, represented as Integers.
     * @throws FileNotFoundException If the provided file does not exist.
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
     * This method calculates the Euclidean distances between each pair of points in
     * the test set and the training set.
     * Each point is represented as a list of floats, and the distance between two
     * points is calculated as the square root of the sum of the squares of the
     * differences of their corresponding features.
     *
     * @param testInput  The test set, represented as a list of points. Each point
     *                   is a list of floats.
     * @param trainInput The training set, represented as a list of points. Each
     *                   point is a list of floats.
     * @return A 2D list of floats, where each inner list represents the distances
     *         from a test point to all training points.
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
     * This method calculates the Euclidean distances between each pair of points in
     * the test set and the training set.
     * Each point is represented as a list of floats, and the distance between two
     * points is calculated as the square root of the sum of the squares of the
     * differences of their corresponding features.
     *
     * @param testInput  The test set, represented as a list of points. Each point
     *                   is a list of floats.
     * @param trainInput The training set, represented as a list of points. Each
     *                   point is a list of floats.
     * @return A 2D list of floats, where each inner list represents the distances
     *         from a test point to all training points.
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
     * @param TEST_LABEL  Test labels
     * @return Accuracy of the predictions
     */
    public static Double calculateAccuracy(List<Integer> predictions, List<Integer> TEST_LABEL) {
        try {
            return (double) IntStream.range(0, TEST_LABEL.size())
                    .filter(i -> TEST_LABEL.get(i).equals(predictions.get(i)))
                    .count() / TEST_LABEL.size() * 100;
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
            int minIndex = innerArr.indexOf(Collections.min(innerArr));
            predictions.add(trainLabels.get(minIndex));
        }
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("output2.txt"))) {
            StringJoiner joiner = new StringJoiner(" ");
            for (Integer prediction : predictions) {
                joiner.add(prediction.toString());
            }
            writer.write(joiner.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }

        return predictions;
    }

    /**
     * Shortcut for calculating the accuracy of the model using the Euclidean
     * 
     * @return Accuracy of the model as a double
     */
    public static Double shortcutEuclidean(List<List<Float>> testSet, List<List<Float>> trainSet,
            List<Integer> testLabel,

            List<Integer> trainLabel) {
        return calculateAccuracy(predictLabel(calculateEuclideanDistances(testSet, trainSet), trainLabel), testLabel);
    }

    /**
     * Shortcut for calculating the accuracy of the model using the Manhattan
     * 
     * @return Accuracy of the model as a double
     */
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
     * This method generates an initial population of binary strings for a genetic
     * algorithm.
     * Each binary string represents a possible selection of features for a
     * k-Nearest Neighbors algorithm.
     * The binary strings are generated randomly.
     *
     * @param initialPopulationSize The size of the initial population to generate.
     * @param length                The length of the binary strings to generate.
     * @return A list of binary strings representing the initial population.
     */
    public static List<String> generateinitialPopulation(int initialPopulationSize) {
        List<String> binaryStrings = new ArrayList<>();
        Random random = new Random();

        for (int i = 0; i < initialPopulationSize; i++) {
            StringBuilder binaryString = new StringBuilder();

            for (int j = 0; j < initalPopulationLength; j++) {
                int randomBit = random.nextInt(2);
                binaryString.append(randomBit);
            }
            binaryStrings.add(binaryString.toString());
        }

        return binaryStrings;
    }

    /**
     * This method finds the indices of '1's in each binary string in a list.
     * Each binary string represents a possible selection of features for a
     * k-Nearest Neighbors algorithm,
     * where '1' indicates that the corresponding feature should be used, and '0'
     * indicates that it should be ignored.
     *
     * @param parentSet A list of binary strings representing feature selections.
     * @return A 2D list of integers, where each inner list represents the indices
     *         of '1's in a binary string.
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
     * This method retrieves the data corresponding to the '1's in a binary string
     * from a dataset.
     * Each binary string represents a possible selection of features for a
     * k-Nearest Neighbors algorithm,
     * where '1' indicates that the corresponding feature should be used, and '0'
     * indicates that it should be ignored.
     *
     * @param indices A list of integers representing the indices of '1's in a
     *                binary string.
     * @param dataSet The dataset, represented as a 2D list of floats. Each inner
     *                list represents a data point and contains the features of the
     *                point.
     * @return A 2D list of floats, where each inner list represents a data point
     *         and contains the features corresponding to the '1's in the binary
     *         string.
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
     * This method applies a genetic algorithm to optimize the feature selection for
     * a k-Nearest Neighbors algorithm.
     * It generates an initial population of binary strings, where each string
     * represents a possible selection of features,
     * and then iteratively applies crossover and mutation to generate new
     * generations of feature selections.
     * The accuracy of the k-Nearest Neighbors algorithm is calculated for each
     * feature selection,
     * and the feature selections are ranked based on their accuracies.
     * The process continues until a feature selection achieves the expected
     * accuracy or the generation limit is reached.
     *
     * @param generationLimit         The maximum number of generations to generate.
     * @param expectedAccuracy        The expected accuracy of the k-Nearest
     *                                Neighbors algorithm.
     * @param initialPopulationSize   The size of the initial population of feature
     *                                selections.
     * @param mutationChance          The chance of a mutation occurring during the
     *                                generation of a new population.
     * @param testSet                 The test set, represented as a list of points.
     *                                Each point is a list of floats.
     * @param trainSet                The training set, represented as a list of
     *                                points. Each point is a list of floats.
     * @param testLabel               The labels for the test set.
     * @param trainLabel              The labels for the training set.
     * @param initialPopulationLength The length of the binary strings in the
     *                                initial population.
     * @return The binary string representing the best feature selection found by
     *         the genetic algorithm.
     */
    public static String calculateGeneticAlgorithm(double expectedAccuracy,
            int initialPopulationSize,
            int mutationChance) {

        final List<List<Float>> localTestSet = new ArrayList<>(TEST_DATA);
        final List<List<Float>> localTrainSet = new ArrayList<>(TRAIN_DATA);

        List<String> parentSet = generateinitialPopulation(initialPopulationSize);

        List<String> mutatedParentSet = new ArrayList<>(parentSet);
    
        int c = 0;

        // STORES THE ACCURACY OF EVERY PARENT
        List<Double> resultSet = new ArrayList<>();

        while (resultSet.isEmpty() || resultSet.get(0) < expectedAccuracy) {
            c++;
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

                double result = shortcutEuclidean(modifiedTestSet, modifiedTrainSet, TEST_LABEL, TRAIN_LABEL);

                accuracyMap.put(mutatedParentSet.get(i), result);
                resultSet.add(result);
            }
            accuracyMap.size();
            mutatedParentSet.clear();
            resultSet.sort(Comparator.reverseOrder());
            accuracyMap = accuracyMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (oldValue, newValue) -> oldValue,
                            LinkedHashMap::new));

            List<String> mostAccurateParents = new ArrayList<>(accuracyMap.keySet());

            mutatedParentSet = mostAccurateParents.subList(0, mostAccurateParents.size() / 2);

            if (mutatedParentSet.size() != initialPopulationSize
                    && initialPopulationSize - mutatedParentSet.size() >= 2) {
                List<String> addition = mutator(crossover(generateinitialPopulation(
                        initialPopulationSize - mutatedParentSet.size())), mutationChance);
                for (int i = 0; i < addition.size(); i++) {
                    mutatedParentSet.add(addition.get(i));
                }
            }
            System.out.println("Generation: " + c + " Accuracy: " + resultSet.get(0));
        }
        System.out.println("Found best accuracy: " + resultSet.get(0) + " with parent " + mutatedParentSet.get(0));

        return mutatedParentSet.get(0);

    }

    /**
     * This method performs crossover on a set of binary strings representing
     * feature selections for a k-Nearest Neighbors algorithm.
     * Crossover is a genetic algorithm operation where two parent strings are
     * selected and parts of their data are swapped to create new offspring.
     * The method performs crossover on pairs of binary strings in the set, with the
     * first string in each pair being the first string in the set and the second
     * string being another string in the set.
     * The points at which the data is swapped are determined randomly.
     *
     * @param parentSet A list of binary strings representing feature selections.
     * @return A list of binary strings representing the original feature selections
     *         and the new feature selections created by crossover.
     */
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
     * This method performs mutation on a set of binary strings representing feature
     * selections for a k-Nearest Neighbors algorithm.
     * Mutation is a genetic algorithm operation where a small random change is made
     * to the data to maintain diversity in the population.
     * The method performs mutation on each binary string in the set, with each bit
     * in the string having a chance to be flipped.
     * The chance of a bit being flipped is determined by the mutationChance
     * parameter.
     *
     * @param parentSet      A list of binary strings representing feature
     *                       selections.
     * @param mutationChance The chance of a bit being flipped, represented as an
     *                       integer between 0 and 100.
     * @return A list of binary strings representing the mutated feature selections.
     */
    public static List<String> mutator(List<String> parentSet, int mutationChance) {

        List<String> modifiableParentSet = new ArrayList<>(parentSet);
        List<String> mutationDone = new ArrayList<>();

        mutationDone.add(modifiableParentSet.get(0));

        Random rand = new Random();
        for (int i = 1; i < modifiableParentSet.size() - 1; i++) {

            StringBuilder p1 = new StringBuilder(modifiableParentSet.get(i));

            for (int j = 0; j < p1.length(); j++) {
                if (rand.nextInt(100) < mutationChance) {
                    p1.setCharAt(j, (p1.charAt(j) == '0') ? '1' : '0');
                }
            }

            mutationDone.add(p1.toString());
        }

        return mutationDone;
    }

}