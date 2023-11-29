import java.util.ArrayList;
import java.util.Arrays;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class kNN1 {

    public static void main(String[] args) throws FileNotFoundException {
        parseData();
        parseLabel();
    }

    /*
     * Parses String data to Float and creates a 2D ArrayList
     * 
     * @return 2D ArrayList containing data parsed to float
     */
    public static ArrayList<ArrayList<Float>> parseData() throws FileNotFoundException {

        ArrayList<ArrayList<Float>> dataArray = new ArrayList<>();
        File dataFile = new File("train_data.txt");
        Scanner dataScanner = new Scanner(dataFile);

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
    public static ArrayList<Integer> parseLabel() throws FileNotFoundException {
        ArrayList<Integer> labelArr = new ArrayList<>();
        File labelFile = new File("train_label.txt");
        Scanner labelReader = new Scanner(labelFile);
        while (labelReader.hasNext()) {
            labelArr.add(Integer.parseInt(labelReader.next()));
        }
        labelReader.close();
        return labelArr;
    }









    
}
