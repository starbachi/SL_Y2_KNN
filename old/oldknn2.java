package old;
    
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.stream.IntStream;

public class oldknn2 {

    
    public static void main(String[] args) throws FileNotFoundException {

        



    

    }




    

    /**
     * Calculates the Euclidean distance of every test pattern to all training patterns
     * 
     * @param testInput Test Patterns
     * @param trainInput Train Patterns
     * @return 2D List of Euclidean distances
     */
    private static List<List<Double>> calculateEuclidean(List<List<Float>> testInput,List<List<Float>> trainInput) {
        List<List<Double>> distances = new ArrayList<>();
        List<Double> innerArr;
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
     * Calculates the Manhattan distance of every test pattern to all training patterns
     * 
     * @param testInput Test Patterns
     * @param trainInput Train Patterns
     * @return 2D ArrayList of Manhattan distances
     */
    private static List<List<Double>> calculateManhattan(List<List<Float>> testInput,List<List<Float>> trainInput) {
        List<List<Double>> distances = new ArrayList<>();
        List<Double> innerArr;
        
        for (int i = 0; i < testInput.size(); i++) {
            innerArr = new ArrayList<>();
            for (int j = 0; j < trainInput.size(); j++) {
                double sum = 0;
                for (int k = 0; k < testInput.get(i).size(); k++) {
                    sum += Math.abs(testInput.get(i).get(k) - trainInput.get(j).get(k));
                }
                innerArr.add(sum);
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
    public static List<Integer> predictLabel(List<List<Double>> inputArr, List<Integer> labels) {
        List<Integer> predictions = new ArrayList<>();
        for (List<Double> innerArr : inputArr) {
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
    public static Double calculateAccuracy (List<Integer> predictions, List<Integer> actuals) {
        return (double) IntStream.range(0, actuals.size()).filter(i -> actuals.get(i).equals(predictions.get(i))).count() / actuals.size() * 100;
    }




    public static List<List<Float>> removeTopColumns(List<List<Float>> trainSet, double threshold, int numColumnsToRemove) {
        // Count the number of values under the threshold for each column
        List<Integer> columnCounts = new ArrayList<>();
        for (int j = 0; j < trainSet.get(0).size(); j++) {
            int count = 0;
            for (List<Float> row : trainSet) {
                if (row.get(j) < threshold) {
                    count++;
                }
            }
            columnCounts.add(count);
        }

        // Create a list of column indices and sort them based on counts
        List<Integer> columnIndices = new ArrayList<>();
        for (int i = 0; i < columnCounts.size(); i++) {
            columnIndices.add(i);
        }

        Collections.sort(columnIndices, Comparator.comparingInt(columnCounts::get));

        // Create a new ArrayList to store the data without removed columns
        List<List<Float>> newDataSet = new ArrayList<>();

        // Add only the columns that are not removed
        for (List<Float> row : trainSet) {
            List<Float> newRow = new ArrayList<>();
            for (int i = 0; i < row.size(); i++) {
                if (!columnIndices.subList(0, numColumnsToRemove).contains(i)) {
                    newRow.add(row.get(i));
                }
            }
            newDataSet.add(newRow);
        }

        return newDataSet;
    }

    public static void writeOutputToFile(String fileName, List<List<Float>> dataSet) {
        try (FileWriter writer = new FileWriter(fileName)) {
            for (List<Float> row : dataSet) {
                for (float value : row) {
                    writer.write(value + " ");
                }
                writer.write(System.lineSeparator());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }




}
