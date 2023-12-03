import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.stream.IntStream;


/**
 * @author Erdem Elik
 * @version 1
 *
 */

public class kNN1 {
    
    public static void main(String[] args) throws FileNotFoundException {
        ArrayList<ArrayList<Float>> trainDataArray = new ArrayList<>(parseData(new File("train_data.txt")));
        ArrayList<ArrayList<Float>> testDataArray = new ArrayList<>(parseData(new File("test_data.txt")));
        ArrayList<Integer> trainLabelArray = new ArrayList<>(parseLabel(new File("train_label.txt")));
        ArrayList<Integer> testLabelArray = new ArrayList<>(parseLabel(new File("test_label.txt")));
        ArrayList<ArrayList<Double>> euclideanDistances = new ArrayList<>(calculateEuclidean(testDataArray, trainDataArray));
        System.out.println(calculateAccuracy(predictLabel(euclideanDistances,trainLabelArray),testLabelArray).toString());
    }

    /**
     * Parses file to float and creates a 2D ArrayList of features
     * @param file Train and test data to parse
     */
    public static ArrayList<ArrayList<Float>> parseData(File file) throws FileNotFoundException {

        ArrayList<ArrayList<Float>> dataArray = new ArrayList<>();
        Scanner dataScanner = new Scanner(file);
        while (dataScanner.hasNextLine()) {
            dataArray.add(new ArrayList<>(Arrays.asList(
                    Arrays.stream(dataScanner.nextLine().split(" ")).map(Float::valueOf).toArray(Float[]::new))));
        }
        dataScanner.close();

        return dataArray;
    }

    /**
     * Parses String labels to Integer and creates an ArrayList<Integer>
     * 
     * @return An ArrayList<Integer> containing labels consisting of 0 and 1
     */
    public static ArrayList<Integer> parseLabel(File file) throws FileNotFoundException {
        ArrayList<Integer> labelArr = new ArrayList<>();
        Scanner labelReader = new Scanner(file);
        while (labelReader.hasNext()) {
            labelArr.add(Integer.parseInt(labelReader.next()));
        }
        labelReader.close();
        return labelArr;
    }

    /**
     * Calculates the Euclidean distance of every test pattern to all training patterns
     * @param testInput Test Patterns
     * @param trainInput Train Patterns
     */
    private static ArrayList<ArrayList<Double>> calculateEuclidean(ArrayList<ArrayList<Float>> testInput,ArrayList<ArrayList<Float>> trainInput) {
        ArrayList<ArrayList<Double>> distances = new ArrayList<>();
        ArrayList<Double> innerArr;
        for (int i = 0; i < testInput.size(); i++) {
            innerArr = new ArrayList<>();
            for (int j = 0; j < trainInput.size(); j++) {
                double sum = 0;
                for (int k = 0; k < testInput.get(i).size(); k++) {
                    sum += Math.pow(testInput.get(i).get(k) - trainInput.get(j).get(k), 2);
                }
                innerArr.add(Math.sqrt(sum));
            }
            distances.add(innerArr);
        }
        return distances;
    }

    /**
     * Finds the label of the nearest neighbour
     * @param inputArr ArrayList containing calculated distances
     * @param labels ArrayList of labels
     * @return ArrayList of predictions
     */
    public static ArrayList<Integer> predictLabel(ArrayList<ArrayList<Double>> inputArr, ArrayList<Integer> labels) {
        ArrayList<Integer> predictions = new ArrayList<>();
        for (ArrayList<Double> innerArr : inputArr) {
            int minIndex = labels.get(innerArr.indexOf(Collections.min(innerArr)));
            predictions.add(minIndex);
        }
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("output1.txt"))) {
            writer.write(predictions.toString());
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        return predictions;

    }

    /**
     * Compares predictions and actual labels to calculate the accuracy of the model 
     * 
     * @param predictions Prediction labels
     * @param actuals Test labels
     * @return Accuracy of the predictions
     */
    public static Double calculateAccuracy (ArrayList<Integer> predictions, ArrayList<Integer> actuals) {
        return (double) IntStream.range(0, actuals.size()).filter(i -> actuals.get(i).equals(predictions.get(i))).count() / actuals.size() * 100;
    }

}
