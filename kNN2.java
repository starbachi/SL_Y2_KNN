import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.IntStream;

public class kNN2 {

    public static void main(String[] args) throws FileNotFoundException {

        // Data Parsed
        List<List<Float>> trainData = new ArrayList<>(parseData(new File("train_data.txt")));
        List<List<Float>> testData = new ArrayList<>(parseData(new File("test_data.txt")));

        // Label Parsed
        List<Integer> trainLabel = new ArrayList<>(parseLabel(new File("train_label.txt")));
        List<Integer> testLabel = new ArrayList<>(parseLabel(new File("test_label.txt")));

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

        //Delete later
        // System.out.println(calculateEuclideanDistances(testData, trainPairsMap));
        // saveDistancesToFile(calculateEuclideanDistances(testData, trainPairsMap), "abc.txt");

        //Stores predicted labels
        List<Integer> euclideanPredictedLabels = new ArrayList<>(predictLabel(calculateEuclideanDistances(testData, trainPairsMap), trainLabel));
        List<Integer> manhattanPredictedLabels = new ArrayList<>(predictLabel(calculateManhattanDistances(testData, trainPairsMap), trainLabel));


        System.out.println("Euclidean " + calculateAccuracy(euclideanPredictedLabels, testLabel) + "%");
        System.out.println("Manhattan " + calculateAccuracy(manhattanPredictedLabels, testLabel) + "%");
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
                    innerArr.add((float)Math.sqrt(sum));
                }
            }
            distances.add(innerArr);
        }
        return distances;
    }

    /**
     * Calculates the Manhattan distance of every test pattern to all training patterns
     * 
     * @param testPatterns 2D List of test patterns
     * @param trainMap Map of the train patterns
     * @return 2D List of Manhattan distances
     */
    private static List<List<Float>> calculateManhattanDistances(List<List<Float>> testPatterns,Map<Integer, List<List<Float>>> trainMap) {
        List<List<Float>> distances = new ArrayList<>();
        for (int i = 0; i < testPatterns.size(); i++) {
            List<Float> innerArr = new ArrayList<>();
            for (int j = 0; j < trainMap.size(); j++) {
                for (int k = 0; k < trainMap.get(j).size(); k++) {
                    double sum = 0;
                    for (int l = 0; l < testPatterns.get(i).size(); l++) {
                        sum += Math.abs(testPatterns.get(i).get(l) - trainMap.get(j).get(k).get(l));
                    }
                    innerArr.add((float)Math.sqrt(sum));
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
     * @param actuals Test labels
     * @return Accuracy of the predictions
     */
    public static Double calculateAccuracy (List<Integer> predictions, List<Integer> actuals) {
        return (double) IntStream.range(0, actuals.size()).filter(i -> actuals.get(i).equals(predictions.get(i))).count() / actuals.size() * 100;
    }

    /////////////////////////////////////////////////////
    //////////////// END OF CALCULATIONS ////////////////
    /////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////
    //////////////// PREDICTIONS & FILE PRINTING ////////////////
    /////////////////////////////////////////////////////////////

    /**
     * Finds the label of the nearest neighbour
     * @param inputToPredict List containing distances
     * @param trainLabels List of TRAINING labels
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
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        return predictions;

    }


    



    
    // private static void saveDistancesToFile(List<List<Float>> distances, String filename) {
    //     Path filePath = Paths.get(filename);

    //     try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath.toFile()))) {
    //         for (List<Float> row : distances) {
    //             writer.write(row.toString());
    //             writer.newLine();
    //         }
    //     } catch (IOException e) {
    //         e.printStackTrace();
    //     }
    // }



}
