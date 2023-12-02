import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.text.DecimalFormat;
import java.util.Scanner;

public class kNN1 {

    public static void main(String[] args) throws FileNotFoundException {
        ArrayList<ArrayList<Float>> trainingDataArray = new ArrayList<>(parseData(new File("train_data.txt")));
        ArrayList<ArrayList<Float>> testDataArray = new ArrayList<>(parseData(new File("test_data.txt")));
        ArrayList<Integer> trainingLabelArray = new ArrayList<>(parseLabel(new File("train_label.txt")));
        ArrayList<Integer> testLabelArray = new ArrayList<>(parseLabel(new File("test_label.txt")));
        ArrayList<Double> euclideanDistances = new ArrayList<>(calculateEuclidean(testDataArray, trainingDataArray, "output.txt"));
        System.out.println(euclideanDistances.toString());
    }

    /*
     * Parses @param file to float and creates a 2D ArrayList of features
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

    /*
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

    private static ArrayList<Double> calculateEuclidean(ArrayList<ArrayList<Float>> testData, ArrayList<ArrayList<Float>> trainingData, String outputFileName) {
        ArrayList<Double> distances = new ArrayList<>();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName))) {
            double sum = 0;
            
            for (int i = 0; i < testData.size(); i++) {
                for (int k = 0; k < testData.get(i).size(); k++) {
                    sum +=  Math.pow(testData.get(i).get(k) - trainingData.get(i).get(k), 2);
                }
                distances.add(sum = Math.sqrt(sum));
                writer.write(String.valueOf(sum));
                writer.newLine();
            }
            

           
        }
        catch (IOException e){
            e.printStackTrace();
        }
        return distances;
    }

}
