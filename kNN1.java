import java.util.ArrayList;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;



public class kNN1 {

    public static void main(String[] args) throws FileNotFoundException {
        readFile();
    }




    public static ArrayList<String> readFile() throws FileNotFoundException {

        ArrayList<String> dataArr = new ArrayList<String>();
        File dataFile = new File("train_data.txt");
        
        Scanner dataReader = new Scanner(dataFile);

        while (dataReader.hasNextLine()) {
            dataArr.add(dataReader.nextLine());
        }
        dataReader.close();

        ArrayList<String> labelArr = new ArrayList<String>();
        File labelFile = new File("train_label.txt");
        
        Scanner labelReader = new Scanner(labelFile);

        while (labelReader.hasNext()) {
            labelArr.add(labelReader.next());
            System.out.println(labelReader.next());
        }
        labelReader.close();



        return dataArr;
    }
}





    
