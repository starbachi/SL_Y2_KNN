import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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

        // Will contain labels and data paired
        Map<Integer, List<List<Float>>> trainPairsMap = new HashMap<>();
        Map<Integer, List<List<Float>>> testPairsMap = new HashMap<>();

        // Simply contains all "0" labeled trainData in index 0 and all "1" labeled in
        // index 1
        List<List<List<Float>>> trainDataGrouped = groupAllData(trainLabel, trainData);
        trainPairsMap.put(0, trainDataGrouped.get(0));
        trainPairsMap.put(1, trainDataGrouped.get(1));

        // Simply contains all "0" labeled testData in index 0 and all "1" labeled in
        // index 1
        List<List<List<Float>>> testDataGrouped = groupAllData(testLabel, testData);
        testPairsMap.put(0, testDataGrouped.get(0));
        testPairsMap.put(1, testDataGrouped.get(1));

        // Delete later
        // System.out.println(calculateEuclideanDistances(testData, trainPairsMap));
        // saveDistancesToFile(calculateEuclideanDistances(testData, trainPairsMap),
        // "abc.txt");

        // Stores predicted labels
        List<Integer> euclideanPredictedLabels = new ArrayList<>(
                predictLabel(calculateEuclideanDistances(testData, trainPairsMap), trainLabel));
        List<Integer> manhattanPredictedLabels = new ArrayList<>(
                predictLabel(calculateManhattanDistances(testData, trainPairsMap), trainLabel));

        // System.out.println("Euclidean " + calculateAccuracy(euclideanPredictedLabels,
        // testLabel) + "%");
        // System.out.println("Manhattan " + calculateAccuracy(manhattanPredictedLabels,
        // testLabel) + "%");

        String[] binary = generateInitialPopulation(1, testData.get(0).size());
        System.out.println(binary[0]);
        calculateGeneticAlgorithm(testData, trainData, new String[] {binary[0]});

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

    ///////////////////////////////////////////////
    //////////////// DATA GROUPING ////////////////
    ///////////////////////////////////////////////

    /**
     * Iterates through the dataSet, matching each index with the corresponding
     * entry in "labelSet" to determine the appropriate group for adding Lists of
     * floats.
     * //FIXME: HARDCODED
     * 
     * @param labelSet contains the labels
     * @param dataSet  contains the data
     * @return 3D List of float with index 0 being the "0" labeled group and index 1
     *         being the "1" labeled group
     */
    public static List<List<List<Float>>> groupAllData(List<Integer> labelSet, List<List<Float>> dataSet) {
        List<List<Float>> ListOf0 = new ArrayList<>();
        List<List<Float>> ListOf1 = new ArrayList<>();
        List<List<List<Float>>> ListOfBoth = new ArrayList<>();
        for (int i = 0; i < labelSet.size() && i < dataSet.size(); i++) {
            if (labelSet.get(i) == 0)
                ListOf0.add(dataSet.get(i));
            else if (labelSet.get(i) == 1) {
                ListOf1.add(dataSet.get(i));
            }
        }

        ListOfBoth.add(ListOf0);
        ListOfBoth.add(ListOf1);
        return ListOfBoth;
    }

    //////////////////////////////////////////////////////////////
    //////////////// DATA IS READY FOR PROCESSING ////////////////
    //////////////////////////////////////////////////////////////

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
    public static List<List<Float>> calculateEuclideanDistances(List<List<Float>> testInput,
            Map<Integer, List<List<Float>>> trainMap) {
        List<List<Float>> distances = new ArrayList<>();
        for (int i = 0; i < testInput.size(); i++) {
            List<Float> innerArr = new ArrayList<>();
            for (int j = 0; j < trainMap.size(); j++) {
                for (int k = 0; k < trainMap.get(j).size(); k++) {
                    double sum = 0;
                    for (int l = 0; l < testInput.get(i).size(); l++) {
                        sum += Math.pow(testInput.get(i).get(l) - trainMap.get(j).get(k).get(l), 2);
                    }
                    innerArr.add((float) Math.sqrt(sum));
                }
            }
            distances.add(innerArr);
        }
        return distances;
    }

    /**
     * Calculates the Manhattan distance of every test pattern to all training
     * patterns
     * 
     * @param testPatterns 2D List of test patterns
     * @param trainMap     Map of the train patterns
     * @return 2D List of Manhattan distances
     */
    private static List<List<Float>> calculateManhattanDistances(List<List<Float>> testPatterns,
            Map<Integer, List<List<Float>>> trainMap) {
        List<List<Float>> distances = new ArrayList<>();
        for (int i = 0; i < testPatterns.size(); i++) {
            List<Float> innerArr = new ArrayList<>();
            for (int j = 0; j < trainMap.size(); j++) {
                for (int k = 0; k < trainMap.get(j).size(); k++) {
                    double sum = 0;
                    for (int l = 0; l < testPatterns.get(i).size(); l++) {
                        sum += Math.abs(testPatterns.get(i).get(l) - trainMap.get(j).get(k).get(l));
                    }
                    innerArr.add((float) Math.sqrt(sum));
                }
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

    /**
     * Generates n-number of random binary strings of length 61 to be used as
     * initial population
     * 
     * @param initialPopulation starting population of the genetic algorithm
     * @return String Array of length initialPopulation containing binary strings of
     *         length 61
     */
    public static String[] generateInitialPopulation(int initialPopulation, int length) {
        String[] binaryStrings = new String[initialPopulation];
        Random random = new Random();

        for (int i = 0; i < initialPopulation; i++) {
            StringBuilder binaryString = new StringBuilder();
            for (int j = 0; j < length; j++) {
                int randomBit = random.nextInt(2);
                binaryString.append(randomBit);
            }
            binaryStrings[i] = binaryString.toString();
        }

        return binaryStrings;
    }

    public static void calculateGeneticAlgorithm(List<List<Float>> testSet, List<List<Float>> trainSet, String[] chromosomeSet) {
        final List<List<Float>> localTestSet = new ArrayList<>(testSet);
        final List<List<Float>> localTrainSet = new ArrayList<>(trainSet);
        List<List<List<Float>>> modifiedTestSet = new ArrayList<>();
        List<List<List<Float>>> modifiedTrainSet = new ArrayList<>();
        List<List<Integer>> indices = new ArrayList<>();

        // Get the indices of 1's in the binary strings
        for (String binaryString : chromosomeSet) {
            List<Integer> innerIndices = new ArrayList<>();
            for (int i = 0; i < binaryString.length(); i++) {
                if (binaryString.charAt(i) == '1') {
                    innerIndices.add(i);
                }
            }
            indices.add(innerIndices);
        }

        List<List<Float>> retrievedTestPattern = new ArrayList<>();
        List<List<Float>> retrievedTrainPattern = new ArrayList<>();


        // Gets the data points at the indices of 1's for modifiedTestData
        for (int i = 0; i < indices.size(); i++) {
            for (int j = 0; j < localTestSet.size(); j++) {
                List<Float> retrievedTestDataPoint = new ArrayList<>();
                for (int indexToLookUp : indices.get(i)) {
                    retrievedTestDataPoint.add(localTestSet.get(j).get(indexToLookUp));
                }
                retrievedTestPattern.add(retrievedTestDataPoint);
            }
            

        }       
        
        // Gets the data points at the indices of 1's for modifiedTrainData
        for (int i = 0; i < indices.size(); i++) {
            for (int j = 0; j < localTrainSet.size(); j++) {
                List<Float> retrievedTrainDataPoint = new ArrayList<>();
                for (int indexToLookUp : indices.get(i)) {
                    retrievedTrainDataPoint.add(localTrainSet.get(j).get(indexToLookUp));
                }
                retrievedTrainPattern.add(retrievedTrainDataPoint);
            }
            
            
        }       

        modifiedTestSet.add(retrievedTestPattern);
        modifiedTrainSet.add(retrievedTrainPattern);

        String filePath = "modifiedTestSet.txt";

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            // Iterate over modifiedTestSet and write each element to the file
            for (List<List<Float>> outerList : modifiedTestSet) {
                for (List<Float> innerList : outerList) {
                    for (Float value : innerList) {
                        writer.write(value + " ");
                    }
                    writer.newLine();
                }
                writer.newLine(); // Add a newline between outer lists
            }

            System.out.println("Data has been written to the file: " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }


        String filePath1 = "modifiedTrainSet.txt";

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath1))) {
            // Iterate over modifiedTestSet and write each element to the file
            for (List<List<Float>> outerList : modifiedTrainSet) {
                for (List<Float> innerList : outerList) {
                    for (Float value : innerList) {
                        writer.write(value + " ");
                    }
                    writer.newLine();
                }
                writer.newLine(); // Add a newline between outer lists
            }

            System.out.println("Data has been written to the file: " + filePath1);
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

}
